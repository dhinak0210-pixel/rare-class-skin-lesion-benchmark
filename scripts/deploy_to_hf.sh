#!/usr/bin/env bash
# scripts/deploy_to_hf.sh
# Automates deployment of the skin lesion classifier demo to Hugging Face Spaces.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/deploy_to_hf.sh <hf-username>/<space-name>"
    echo "Example: bash scripts/deploy_to_hf.sh dhina/skin-lesion-benchmark"
    exit 1
fi

SPACE_REPO="$1"
TEMP_DIR=$(mktemp -d)

echo "========================================================"
echo " Deploying Skin Lesion Decision Support to HF Spaces"
echo " Target Space: https://huggingface.co/spaces/${SPACE_REPO}"
echo "========================================================"

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "[INFO] Initializing git lfs..."
    git lfs install || true
fi

echo "[1/4] Cloning target Hugging Face Space repository..."
git clone "https://huggingface.co/spaces/${SPACE_REPO}" "${TEMP_DIR}"

echo "[2/4] Copying self-contained application assets..."
cp app/app.py "${TEMP_DIR}/"
cp app/requirements.txt "${TEMP_DIR}/"
cp app/README.md "${TEMP_DIR}/"
cp app/model.pt "${TEMP_DIR}/"
mkdir -p "${TEMP_DIR}/samples"
cp -r app/samples/* "${TEMP_DIR}/samples/"

echo "[3/4] Committing assets..."
cd "${TEMP_DIR}"
git add app.py requirements.txt README.md model.pt samples/
git commit -m "Deploy self-contained skin lesion classifier with Grad-CAM and sample cases" || echo "No changes to commit"

echo "[4/4] Pushing to Hugging Face Spaces..."
git push origin main

echo "========================================================"
echo "🎉 Deployment successful!"
echo "Your app is now building and live at:"
echo "https://huggingface.co/spaces/${SPACE_REPO}"
echo "========================================================"

rm -rf "${TEMP_DIR}"

