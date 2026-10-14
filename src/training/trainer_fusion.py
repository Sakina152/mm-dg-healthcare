"""Train/evaluate ECG + age/sex metadata fusion without target-site selection."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ptbxl_dataset import DIAGNOSTIC_SUPERCLASSES, load_ptbxl_metadata
from src.data.ptbxl_multimodal_dataset import PTBXLMetadataDataset
from src.evaluation.metrics import compute_metrics
from src.models.classifier import SignalOnlyClassifier
from src.models.fusion import FusionClassifier
from src.models.metadata_encoder import MetadataEncoder
from src.models.signal_encoder import SignalEncoder
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed, worker_init_fn


def _metrics(y, p, rare_idx):
    return compute_metrics(y, p, class_labels=list(range(len(DIAGNOSTIC_SUPERCLASSES))), rare_class_labels=[rare_idx]).to_dict()


def _predict(model, loader, device, condition="full", mode="fusion"):
    model.eval(); preds=[]; labels=[]
    with torch.no_grad():
        for b in loader:
            signal=b["signal"].to(device); metadata=b["metadata"].to(device); y=b["label"].numpy()
            if mode == "metadata_only": logits=model(metadata)
            elif condition == "signal_only": logits=model(signal=signal, metadata=None)
            elif condition == "metadata_only": logits=model(signal=None, metadata=metadata)
            else: logits=model(signal=signal, metadata=metadata)
            preds.extend(logits.argmax(1).cpu().numpy()); labels.extend(y)
    return np.asarray(labels), np.asarray(preds)


def train(cfg):
    set_seed(cfg.get("seed",42)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    root=Path(cfg.data.ptbxl_dir)
    meta=load_ptbxl_metadata(root)
    train_sites=list(cfg.data.train_sites); test_sites=list(cfg.data.test_sites)
    train_domain=meta[meta.site.isin(train_sites)].reset_index(drop=True)
    target_domain=meta[meta.site.isin(test_sites)].reset_index(drop=True)
    if train_domain.empty or target_domain.empty: raise ValueError("Train or target domain is empty; check site IDs and dataset path.")
    # Split only source-site rows. The target site is not used for checkpoint selection.
    train_df, val_df = train_test_split(train_domain, test_size=float(cfg.train.get("validation_fraction",0.15)),
        random_state=int(cfg.get("seed",42)), stratify=train_domain["superclass"])
    labels={c:i for i,c in enumerate(DIAGNOSTIC_SUPERCLASSES)}
    # Fit normalization exclusively on source training data.
    train_ds=PTBXLMetadataDataset(train_df,root,labels)
    stats=train_ds.metadata_stats
    val_ds=PTBXLMetadataDataset(val_df,root,labels,stats)
    target_ds=PTBXLMetadataDataset(target_domain,root,labels,stats)
    nw=int(cfg.train.get("num_workers",2)); bs=int(cfg.train.batch_size)
    train_loader=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=nw,worker_init_fn=worker_init_fn,drop_last=False)
    val_loader=DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=nw)
    target_loader=DataLoader(target_ds,batch_size=bs,shuffle=False,num_workers=nw)
    embed=int(cfg.model.get("signal_embed_dim",128)); widths=tuple(cfg.model.get("channel_widths",[32,64,128,128]))
    signal_encoder=SignalEncoder(in_channels=12,embed_dim=embed,channel_widths=widths)
    metadata_encoder=MetadataEncoder(input_dim=2,embed_dim=int(cfg.model.get("metadata_embed_dim",128)),hidden_dim=int(cfg.model.get("metadata_hidden_dim",32)))
    model=FusionClassifier(signal_encoder,metadata_encoder,cfg.model.get("fusion_strategy","concat"),len(labels),embed,
        int(cfg.model.get("metadata_embed_dim",128)),int(cfg.model.get("hidden_dim",64))).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(cfg.train.lr),weight_decay=float(cfg.train.get("weight_decay",0.0)))
    criterion=torch.nn.CrossEntropyLoss()
    out=Path(cfg.output.run_dir); out.mkdir(parents=True,exist_ok=True); save_config(cfg,out/"config_used.yaml")
    with open(out/"metadata_normalization.json","w") as f: json.dump(stats,f,indent=2)
    rare_name=str(cfg.rare_class.get("name","HYP")); rare_idx=labels[rare_name]
    best=-1.0; history=[]
    p_drop_signal=float(cfg.get("modality_dropout",{}).get("p_drop_signal",0.0)) if cfg.get("modality_dropout",{}).get("enabled",False) else 0.0
    p_drop_meta=float(cfg.get("modality_dropout",{}).get("p_drop_metadata",0.0)) if cfg.get("modality_dropout",{}).get("enabled",False) else 0.0
    if p_drop_signal+p_drop_meta >= 1: raise ValueError("Modality dropout probabilities must sum to < 1 so at least one modality remains.")
    for epoch in range(int(cfg.train.epochs)):
        t0=time.time(); model.train(); loss_sum=0.; seen=0
        for b in tqdm(train_loader,leave=False):
            s=b["signal"].to(device); m=b["metadata"].to(device); y=b["label"].to(device); n=y.size(0)
            ds=torch.zeros(n,dtype=torch.bool,device=device); dm=torch.zeros_like(ds)
            if p_drop_signal or p_drop_meta:
                r=torch.rand(n,device=device); ds=r < p_drop_signal; dm=(r >= p_drop_signal) & (r < p_drop_signal+p_drop_meta)
            logits=model(signal=s,metadata=m,drop_signal_mask=ds,drop_metadata_mask=dm)
            loss=criterion(logits,y); optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()*n; seen += n
        vy,vp=_predict(model,val_loader,device)
        vm=_metrics(vy,vp,rare_idx)
        record={"epoch":epoch+1,"train_loss":loss_sum/max(seen,1),"validation_metrics":vm,"elapsed_sec":time.time()-t0}
        history.append(record); print(f"epoch {epoch+1}/{cfg.train.epochs} loss={record['train_loss']:.4f} val_rare_f1={vm['rare_class_f1']:.4f} val_macro_f1={vm['macro_f1']:.4f}")
        if vm["rare_class_f1"] > best:
            best=vm["rare_class_f1"]
            torch.save({"model_state_dict":model.state_dict(),"metadata_stats":stats,"label_to_idx":labels,"epoch":epoch+1,"validation_metrics":vm},out/"best_model.pt")
    with open(out/"history.json","w") as f: json.dump(history,f,indent=2)
    # Reload best source-validation checkpoint before touching target test results.
    ckpt=torch.load(out/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(ckpt["model_state_dict"])
    target_results={}
    for condition in ["full","signal_only","metadata_only"]:
        y,p=_predict(model,target_loader,device,condition)
        target_results[condition]=_metrics(y,p,rare_idx)
    # Metadata-only baseline is separately trained, with source-only validation selection.
    meta_model=MetadataEncoder(input_dim=2,embed_dim=64,hidden_dim=32).to(device)
    meta_head=torch.nn.Linear(64,len(labels)).to(device)
    meta_opt=torch.optim.Adam(list(meta_model.parameters())+list(meta_head.parameters()),lr=float(cfg.train.lr),weight_decay=float(cfg.train.get("weight_decay",0.0)))
    best_meta=-1.; best_meta_state=None
    for _ in range(int(cfg.train.epochs)):
        meta_model.train(); meta_head.train()
        for b in train_loader:
            m=b["metadata"].to(device); y=b["label"].to(device); logits=meta_head(meta_model(m)); loss=criterion(logits,y)
            meta_opt.zero_grad(); loss.backward(); meta_opt.step()
        meta_model.eval(); meta_head.eval(); yp=[]; yy=[]
        with torch.no_grad():
            for b in val_loader:
                logits=meta_head(meta_model(b["metadata"].to(device))); yp.extend(logits.argmax(1).cpu().numpy()); yy.extend(b["label"].numpy())
        score=_metrics(np.asarray(yy),np.asarray(yp),rare_idx)["rare_class_f1"]
        if score>best_meta:
            best_meta=score; best_meta_state={"encoder":meta_model.state_dict(),"head":meta_head.state_dict()}
    meta_model.load_state_dict(best_meta_state["encoder"]); meta_head.load_state_dict(best_meta_state["head"]); meta_model.eval(); meta_head.eval()
    yy=[]; yp=[]
    with torch.no_grad():
        for b in target_loader:
            logits=meta_head(meta_model(b["metadata"].to(device))); yp.extend(logits.argmax(1).cpu().numpy()); yy.extend(b["label"].numpy())
    target_results["metadata_only_baseline"]=_metrics(np.asarray(yy),np.asarray(yp),rare_idx)
    result={"rare_class":rare_name,"train_sites":train_sites,"validation_source_site_rows":len(val_df),"target_site_rows":len(target_domain),"checkpoint_epoch":ckpt["epoch"],"validation_metrics_at_checkpoint":ckpt["validation_metrics"],"target_metrics":target_results,"metadata_features":["age","sex"],"note":"Target site was evaluated only after checkpoint selection; diagnostic codes/reports were not used as inputs."}
    with open(out/"metrics.json","w") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2)); print(f"Results written to {out}")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); args=p.parse_args(); train(load_config(args.config))

if __name__=="__main__": main()
