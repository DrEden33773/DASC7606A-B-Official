# ConvMixer Quick Fix Guide

## What Was Fixed

### Problem

- ❌ Validation accuracy stuck at 1%
- ❌ Validation loss continuously increasing
- ❌ Loss explosion during training

### Root Causes

1. **Label smoothing was disabled** when using mixup+cutmix (wrong for ConvMixer)
2. **Adaptive augmentation** was designed for ResNet/WRN (doesn't suit ConvMixer's architecture)

### Solutions

1. ✅ **Enable label smoothing for ConvMixer** (even with mixup/cutmix)
2. ✅ **Uniform 50-50 augmentation** (Mixup vs CutMix)

## Quick Start

### Recommended: ConvMixer-1024/20 (Memory Efficient)

```bash
python main.py \
    --model convmixer_1024_20 \
    --optimizer adamw \
    --lr 0.001 \
    --weight_decay 0.01 \
    --batch_size 128 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --cutmix_alpha 1.0 \
    --mixup_alpha 0.5 \
    --randaugment_n 2 \
    --randaugment_m 10 \
    --label_smoothing 0.1 \
    --drop_path_rate 0.1 \
    --amp \
    --use_ema
```

**Or use the script:**

```bash
./temp/convmixer_1024_20_fixed.ps1
```

### Alternative: ConvMixer-768/32 (Higher Accuracy Potential)

```bash
./temp/convmixer_768_32_fixed.ps1
```

⚠️ Requires ~12GB memory, batch size reduced to 64

## Expected Results

### First 10 Epochs (Validation Check)

- ✅ Val acc should **increase steadily** (not stuck at 1%)
- ✅ Val loss should decrease or stay stable (not explode)
- ✅ Train acc: 1% → 15-25%

### After Full Training (200 epochs)

| Model | Memory | Speed | Target F1 |
|-------|--------|-------|-----------|
| 1024/20 | 4-6GB | Fast | 0.83-0.86 |
| 768/32 | ~12GB | Slower | 0.84-0.87 |

## Key Changes

### 1. Uniform Augmentation (New Function)

```python
# scripts/train_utils.py:284
def convmixer_augmentation(...):
    """50-50 Mixup/CutMix for ConvMixer"""
    if np.random.rand() < 0.5:
        return mixup_data(...)  # 50%
    else:
        return cutmix_data(...)  # 50%
```

### 2. Label Smoothing Always Enabled

```python
# main.py:618
if args.model.startswith("convmixer"):
    # Label smoothing stays active (0.1 by default)
    logger.info("Label smoothing: {:.2f} (ENABLED)")
```

### 3. Train Epoch Updated

```python
# scripts/train_utils.py:1776
def train_epoch(..., model_name: str = ""):
    if model_name.startswith("convmixer"):
        # Use uniform augmentation
    else:
        # Use adaptive augmentation (for WRN/ResNet)
```

## Troubleshooting

### If val acc still stuck at 1%

1. Check log shows: `"Using UNIFORM augmentation strategy for ConvMixer"`
2. Check log shows: `"Label smoothing: 0.10 (ENABLED)"`
3. Verify you're using the updated code (run `git status` to check)

### If F1 < 0.85 after 200 epochs

Try stronger augmentation:

```bash
--mixup_alpha 0.8 \
--randaugment_m 15 \
--drop_path_rate 0.15
```

## Files Modified

- ✅ `scripts/train_utils.py`: Added `convmixer_augmentation()`, updated `train_epoch()`
- ✅ `main.py`: Fixed label smoothing logic, added model name passing
- ✅ No breaking changes to existing models (WRN, ResNet, PyramidNet)

---
**Status**: ✅ Ready to test  
**Branch**: `exp-6-convmixer`  
**Date**: 2025-10-26
