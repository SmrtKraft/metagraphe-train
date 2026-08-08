#!/usr/bin/env bash
# ── METAGRAPHE-TRAIN cloud bootstrap ────────────────────────────────────────
# Paste-and-run on a fresh CUDA GPU pod (RunPod / Vast / Lambda; Ubuntu + NVIDIA
# driver + a recent PyTorch already present in the standard "PyTorch" template).
# Re-downloads the PUBLIC datasets on the pod (no upload from your Mac), packs
# them, and trains the general transcription model on the GPU. When it finishes,
# checkpoints/ holds the trained model — download the best one back.
#
#   bash bootstrap.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== deps =="
pip install -q --upgrade pip
# torch/torchaudio usually preinstalled on the pod's CUDA image; only add ours
pip install -q torchlibrosa librosa h5py mido mir_eval pandas soundfile "huggingface_hub==0.34.4"

echo "== vendored arch (Apache-2.0) =="
[ -d vendor/piano_transcription ] || git clone -q --depth 1 \
  https://github.com/bytedance/piano_transcription vendor/piano_transcription

mkdir -p data/guitarset checkpoints

echo "== pretrained checkpoint =="
[ -f checkpoints/note_model.pth ] || curl -sL -o checkpoints/note_model.pth \
  "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"

echo "== GuitarSet (Zenodo, CC-BY) =="
if [ ! -d data/guitarset/audio ]; then
  curl -sL -o data/guitarset/annotation.zip "https://zenodo.org/record/3371780/files/annotation.zip?download=1"
  curl -sL -o data/guitarset/audio_mono-mic.zip "https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1"
  ( cd data/guitarset && unzip -q -o annotation.zip -d annotation && unzip -q -o audio_mono-mic.zip -d audio )
fi

echo "== GAPS (HuggingFace, MIT) =="
[ -d data/gaps/audio ] || python -c "from huggingface_hub import snapshot_download; snapshot_download('xavriley/GAPS', repo_type='dataset', local_dir='data/gaps')"

echo "== pack =="
[ -f data/packed/guitarset_train.h5 ] || python pack_guitarset.py
[ -f data/packed/gaps_train.h5 ]      || python pack_gaps.py

echo "== TRAIN (cuda) =="
# GPU fits a much bigger batch; full run. Adjust --epochs to taste.
python -u train.py --device cuda --epochs 50 --batch-size 16 --num-workers 8 \
  --data data/packed 2>&1 | tee runs/cloud_train.log

echo "== DONE =="
ls -t checkpoints/ft_*.pth | head -3
echo "Download the newest checkpoints/ft_*.pth back to your Mac."
