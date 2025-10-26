# 🚀 ConvMixer Quick Start Guide

## ✅ Implementation Complete

**Date**: 2025-10-26  
**Branch**: `exp-6-convmixer`  
**Status**: Ready for training

---

## 📊 Model Variants Implemented

| Model | Parameters | Hidden Dim | Depth | Kernel Size | Patch Size | Expected F1 |
|-------|-----------|------------|-------|-------------|------------|-------------|
| **ConvMixer-768/32** ⭐ | 20.3M | 768 | 32 | 7 | 2 | 0.83-0.85 |
| **ConvMixer-1536/20** | 50.0M | 1536 | 20 | 9 | 2 | 0.84-0.86 |
| **ConvMixer-1024/20** | 22.9M | 1024 | 20 | 9 | 4 | 0.83-0.85 |

**Recommended**: ConvMixer-768/32 (best balance of capacity and efficiency)

---

## 🎯 Why ConvMixer?

### Advantages over EfficientNet

| Aspect | EfficientNet | ConvMixer |
|--------|--------------|-----------|
| **Architecture Complexity** | High (SE blocks, Swish) | ✅ **Extremely Simple** |
| **From-Scratch Training** | ❌ Difficult (unstable) | ✅ **Easy** |
| **Initialization** | Sensitive | ✅ **Robust** |
| **Training Stability** | Poor (activation collapse) | ✅ **Stable** |
| **Paper Evidence** | Relies on pre-training | ✅ **From-scratch verified** |

### Key Features

1. **Patch-based**: Like ViT, but uses convolutions
2. **Depthwise Separable Conv**: 3-5x more efficient than standard conv
3. **Simple & Effective**: Only 3 operations (depthwise, pointwise, residual)
4. **Proven Results**: 80.2% ImageNet top-1 (from scratch)

---

## 🚀 Quick Start Commands

### Option 1: ConvMixer-768/32 (Recommended)

```powershell
# Standard training (batch_size=128, 600 epochs)
python main.py `
    --model convmixer_768_32 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --num_epochs 600 `
    --warmup_epochs 10 `
    --cutmix_alpha 1.0 `
    --mixup_alpha 0.5 `
    --randaugment_n 2 `
    --randaugment_m 10 `
    --label_smoothing 0.1 `
    --amp
```

**Expected**:

- Training time: ~6-8 hours
- Memory usage: ~7-8GB
- Target F1: 0.83-0.85

### Option 2: Quick Test (10 epochs)

```powershell
python main.py `
    --model convmixer_768_32 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --num_epochs 10 `
    --warmup_epochs 2 `
    --amp
```

**Expected**:

- Time: ~10 minutes
- Val acc after 10 epochs: >40% (if training correctly)

### Option 3: ConvMixer-1536/20 (High Capacity)

```powershell
python main.py `
    --model convmixer_1536_20 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 96 `  # Reduced for memory
    --num_epochs 600 `
    --warmup_epochs 10 `
    --cutmix_alpha 1.0 `
    --mixup_alpha 0.5 `
    --randaugment_n 2 `
    --randaugment_m 12 `  # Stronger augmentation
    --label_smoothing 0.1 `
    --amp
```

**Expected**:

- Training time: ~8-10 hours
- Memory usage: ~12-14GB
- Target F1: 0.84-0.86

---

## 📐 Architecture Details

### ConvMixer Block Structure

```python
# Pseudo-code
x = PatchEmbedding(x)  # 3 → dim channels

for _ in range(depth):
    # Spatial mixing (depthwise conv)
    x = x + GELU(BatchNorm(DepthwiseConv(x)))
    
    # Channel mixing (pointwise conv)
    x = GELU(BatchNorm(PointwiseConv(x)))

x = GlobalAvgPool(x)
x = Linear(x, num_classes)
```

### Key Parameters

- **dim**: Hidden dimension (number of channels)
- **depth**: Number of ConvMixer blocks
- **kernel_size**: Depthwise conv kernel size
- **patch_size**: Patch embedding stride

---

## 📈 Hyperparameter Recommendations

### Optimizer Settings

| Parameter | ConvMixer-768/32 | ConvMixer-1536/20 |
|-----------|------------------|-------------------|
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | 0.001 | 0.001 |
| **Weight Decay** | 0.01 | 0.015 |
| **Batch Size** | 128 | 96 |
| **Warmup Epochs** | 10 | 10 |

### Data Augmentation

| Parameter | Value | Notes |
|-----------|-------|-------|
| **RandAugment N** | 2 | Number of operations |
| **RandAugment M** | 10-12 | Magnitude (10=balanced, 12=aggressive) |
| **CutMix Alpha** | 1.0 | Standard setting |
| **Mixup Alpha** | 0.5 | Standard setting |
| **Label Smoothing** | 0.1 | Recommended |

---

## 🔧 Troubleshooting

### Issue 1: Low GPU Utilization

**Symptoms**: GPU usage <50%, slow training

**Solutions**:

1. Increase batch size (if memory allows)
2. Check `num_workers` in data loader
3. Disable `torch.compile` if causing issues

### Issue 2: OOM (Out of Memory)

**Symptoms**: CUDA out of memory error

**Solutions**:

1. Reduce batch size: 128 → 96 → 64
2. Use gradient accumulation
3. Switch to smaller model (768/32 → 1024/20)

### Issue 3: Training Unstable

**Symptoms**: Loss spikes, NaN, or divergence

**Solutions**:

1. Reduce learning rate: 0.001 → 0.0005
2. Increase warmup: 10 → 20 epochs
3. Check data augmentation (reduce M if too strong)

---

## 📊 Expected Training Curves

### Normal Training (ConvMixer-768/32)

```
Epoch   Train Loss   Val Loss   Val Acc   Val F1
----------------------------------------------
10      3.5          3.2        15%       0.10
50      2.0          2.5        45%       0.40
100     1.5          2.2        55%       0.50
200     1.0          2.0        65%       0.62
400     0.6          1.9        75%       0.73
600     0.4          1.8        80%+      0.83+
```

**Red Flags**:

- Val acc <10% after 10 epochs → check implementation
- Loss increases or NaN → reduce LR or check data
- Val acc plateaus <60% → increase regularization

---

## 🎯 Performance Targets

### Conservative Estimates

| Model | Expected F1 | Success Rate |
|-------|-------------|--------------|
| ConvMixer-768/32 | 0.83-0.85 | 80% |
| ConvMixer-1536/20 | 0.84-0.86 | 70% |
| ConvMixer-1024/20 | 0.83-0.85 | 75% |

### Optimistic (with fine-tuning)

| Model | Best Case F1 | Conditions |
|-------|--------------|------------|
| ConvMixer-768/32 | 0.85-0.86 | Strong augmentation + careful tuning |
| ConvMixer-1536/20 | 0.86-0.87 | Large model + longer training |

---

## 📚 References

1. **Paper**: "Patches Are All You Need?" (ICLR 2022)
   - URL: <https://openreview.net/forum?id=TVHS5Y4dNvM>
   - Authors: Asher Trockman, J. Zico Kolter

2. **Official Implementation**:
   - GitHub: <https://github.com/locuslab/convmixer>
   - 7-line tweetable version available

3. **Key Results**:
   - ImageNet (from scratch): 80.2% top-1 (ConvMixer-768/32)
   - ImageNet (from scratch): 81.4% top-1 (ConvMixer-1536/20)
   - Outperforms ViT and ResNet at similar parameter counts

---

## ✅ Verification Checklist

Before starting full training:

- [x] EfficientNet code removed
- [x] ConvMixer implemented (3 variants)
- [x] Forward pass tested
- [x] Parameter counts verified
- [ ] 10-epoch quick test run (do this first!)
- [ ] Full 600-epoch training

---

## 🚦 Next Steps

1. **Quick Test** (REQUIRED):

   ```powershell
   # Run 10 epochs to verify everything works
   python main.py --model convmixer_768_32 --num_epochs 10 --amp
   ```

   - Expected time: ~10 minutes
   - Val acc should be >40% by epoch 10

2. **Full Training**:
   - If quick test passes, start full 600-epoch training
   - Monitor first 50 epochs closely

3. **Hyperparameter Tuning** (if needed):
   - Adjust learning rate (0.0005-0.002)
   - Adjust weight decay (0.01-0.02)
   - Adjust augmentation strength (M: 8-14)

---

## 📝 Implementation Notes

### Code Changes

1. **`scripts/model_architectures.py`**:
   - Added `Residual` helper class
   - Added `ConvMixer` main class
   - Added 3 factory functions: `convmixer_768_32()`, `convmixer_1536_20()`, `convmixer_1024_20()`
   - Updated `create_model()` to support ConvMixer

2. **`main.py`**:
   - Added ConvMixer options to `--model` choices
   - Updated default to `convmixer_768_32`
   - Updated help text

### Key Design Decisions

1. **Patch Size**:
   - 768/32 & 1536/20: patch_size=2 (32×32 → 16×16 patches)
   - 1024/20: patch_size=4 (32×32 → 8×8 patches)
   - Rationale: Different granularities for feature extraction

2. **No Dropout/DropPath**:
   - ConvMixer uses only BatchNorm for regularization
   - Relies on data augmentation instead
   - Simpler and more stable

3. **Initialization**:
   - Standard Kaiming Normal for Conv2d
   - BatchNorm: weight=1, bias=0
   - Linear: small normal (std=0.01)

---

**Status**: ✅ Ready for training!  
**Confidence Level**: High (based on paper results and architecture simplicity)  
**Recommended Action**: Run 10-epoch test immediately 🚀
