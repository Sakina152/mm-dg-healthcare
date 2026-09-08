#!/usr/bin/env bash
# Downloads PTB-XL into data/raw/ptbxl/.
# PTB-XL is fully public on PhysioNet — no login or credentialing needed,
# unlike MIMIC-CXR. ~1.7 GB zipped.
#
# Usage:
#   bash scripts/download_ptbxl.sh
#
# Safe to re-run: skips the download if the target directory already has
# the expected metadata file.

set -euo pipefail

TARGET_DIR="data/raw/ptbxl"
ZIP_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
ZIP_NAME="ptbxl.zip"

if [ -f "${TARGET_DIR}/ptbxl_database.csv" ]; then
  echo "PTB-XL already present at ${TARGET_DIR}/ — skipping download."
  echo "Delete ${TARGET_DIR} and re-run this script to force a fresh download."
  exit 0
fi

mkdir -p "${TARGET_DIR}"
echo "Downloading PTB-XL (~1.7 GB) — this can take a few minutes..."
curl -L -o "${ZIP_NAME}" "${ZIP_URL}"

echo "Extracting..."
unzip -q "${ZIP_NAME}" -d "_ptbxl_extract_tmp"

# The zip extracts to a single versioned top-level folder; flatten it into
# TARGET_DIR so downstream code always sees a stable, version-agnostic path.
INNER_DIR=$(find "_ptbxl_extract_tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
mv "${INNER_DIR}"/* "${TARGET_DIR}/"
rm -rf "_ptbxl_extract_tmp" "${ZIP_NAME}"

echo "Done. PTB-XL is ready at ${TARGET_DIR}/"
echo "Sanity check:"
ls "${TARGET_DIR}" | head -5
