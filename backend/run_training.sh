#!/bin/bash
cd "$(dirname "$0")"
exec ./venv/bin/python -u training/train.py \
  --data_root data/frames \
  --backbone convnext_large \
  --no_vit \
  --use_dino \
  --fusion cross_attention \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --warmup_epochs 2 \
  --patience 5 \
  --num_workers 0
