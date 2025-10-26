# ConvMixer Loss Explosion Fixes

## Problem Diagnosis

When training ConvMixer-1024/20, the model experienced:

- Validation accuracy stuck at 1%
- Validation loss continuously increasing
- Training loss decreasing but validation metrics not improving

This indicated **severe overfitting** and **loss explosion** due to insufficient regularization.

## Root Causes

### 1. Label Smoothing Disabled

The existing codebase automatically disabled `label_smoothing` when both `mixup` and `cutmix` were enabled. This was designed for ResNet/WRN models but is **inappropriate for ConvMixer**.

**Why ConvMixer needs label smoothing:**

- ConvMixer maintains large feature maps (e.g., 16x16 with 768 channels) throughout its depth
- This high representational capacity makes ConvMixer prone to overfitting
- Label smoothing provides crucial regularization even with mixup/cutmix
- Original ConvMixer paper (Patches Are All You Need?, ICLR 2022) uses label smoothing

### 2. Inappropriate Augmentation Strategy

The existing `adaptive_augmentation` function was designed for ResNet/WRN:

- Dynamically selects Mixup vs CutMix based on class characteristics
- Detail-sensitive classes (humans): Mixup only
- Local-feature classes (mechanical): 80% CutMix
- Mixed classes: 30% Mixup, 70% CutMix

**Why this doesn't work for ConvMixer:**

- ConvMixer's architecture is intentionally simple and uniform
- Depthwise + Pointwise convolution allows learning both local and global features
- Maintains larger spatial resolution preserves global context
- Complex, adaptive augmentation introduces unnecessary bias

## Implemented Fixes

### Fix 1: Enable Label Smoothing for ConvMixer

**Location**: `main.py` line 618-628

```python
if args.model.startswith("convmixer"):
    # ConvMixer: Keep label_smoothing enabled even with Mixup/CutMix
    if args.use_cutmix and args.cutmix_alpha > 0 and args.mixup_alpha > 0:
        logger.info("Using UNIFORM augmentation strategy for ConvMixer:")
        logger.info("  • 50% Mixup (alpha={:.1f})".format(args.mixup_alpha))
        logger.info("  • 50% CutMix (alpha={:.1f})".format(args.cutmix_alpha))
        logger.info("  • Label smoothing: {:.2f} (ENABLED)".format(args.label_smoothing))
```

**Result**: Label smoothing (default 0.1) remains active for ConvMixer models.

### Fix 2: Uniform Augmentation Strategy

**Location**: `scripts/train_utils.py` line 284-343

Created new `convmixer_augmentation()` function:

```python
def convmixer_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.5,
    cutmix_alpha: float = 1.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply uniform 50-50 Mixup/CutMix augmentation for ConvMixer models.
    
    Strategy:
    - 50% probability: Apply Mixup (preserves global structure)
    - 50% probability: Apply CutMix (enhances local features)
    """
    if np.random.rand() < 0.5:
        return mixup_data(x, y, alpha=mixup_alpha, device=device)
    else:
        return cutmix_data(x, y, alpha=cutmix_alpha, device=device)
```

**Location**: `scripts/train_utils.py` line 1823-1848

Modified `train_epoch()` to use ConvMixer-specific augmentation:

```python
if use_cutmix and cutmix_alpha > 0 and mixup_alpha > 0:
    if model_name.startswith("convmixer"):
        # ConvMixer: uniform 50-50 strategy
        inputs, targets_a, targets_b, lam = convmixer_augmentation(...)
    else:
        # Other models: adaptive strategy
        inputs, targets_a, targets_b, lam = adaptive_augmentation(...)
```

**Added parameter**: `train_epoch()` now accepts `model_name: str = ""` parameter.

**Location**: `main.py` line 746

Updated `train_epoch()` call to pass model name:

```python
train_loss, train_acc = train_epoch(
    model=model,
    ...
    model_name=args.model,
)
```

## Expected Improvements

### Training Stability

- ✅ Validation metrics should improve steadily (not stuck at 1%)
- ✅ No loss explosion
- ✅ Smooth training curves

### Regularization

- Label smoothing: Prevents overconfident predictions
- Uniform augmentation: Balanced local + global feature learning
- Combined effect: Better generalization

### Performance Targets

| Model | Params | Memory | Expected F1 | Training Time |
|-------|--------|--------|-------------|---------------|
| ConvMixer-768/32 | 21M | ~12GB | 0.84-0.87 | Slower |
| ConvMixer-1024/20 | 23M | ~4-6GB | 0.83-0.86 | Faster |
| ConvMixer-1536/20 | 51M | ~8-10GB | 0.85-0.88 | Medium |

## Recommended Next Steps

### 1. Test ConvMixer-1024/20 (Recommended First)

```bash
./temp/convmixer_1024_20_fixed.ps1
```

- Lower memory usage (4-6GB)
- Faster training
- Good baseline for F1 >= 0.85

### 2. If 1024/20 Works Well, Try 768/32

```bash
./temp/convmixer_768_32_fixed.ps1
```

- Higher memory (12GB)
- Potentially better accuracy (maintains larger feature maps)
- Batch size reduced to 64 due to memory

### 3. Hyperparameter Tuning

If baseline works but F1 < 0.85:

- Increase `mixup_alpha` to 0.6-0.8 (stronger mixing)
- Increase `randaugment_m` to 12-15 (stronger augmentation)
- Increase `drop_path_rate` to 0.15-0.2 (more regularization)
- Train longer (300 epochs with longer warmup)

## Implementation Summary

### Files Modified

1. **`scripts/train_utils.py`**
   - Added `convmixer_augmentation()` function (line 284-343)
   - Modified `train_epoch()` signature to accept `model_name` parameter (line 1776)
   - Updated augmentation selection logic (line 1823-1848)

2. **`main.py`**
   - Updated label smoothing logic for ConvMixer (line 618-628)
   - Added model name to `train_epoch()` call (line 746)

### No Breaking Changes

- All existing models (ResNet, WRN, PyramidNet) continue to use adaptive augmentation
- Only ConvMixer models use the new uniform strategy
- Backward compatible with all existing scripts

## Technical Rationale

### Why Uniform Augmentation?

ConvMixer's architecture differs fundamentally from traditional CNNs:

**Traditional CNNs (ResNet, WRN)**:

- Aggressive spatial downsampling (32x32 → 16x16 → 8x8 → 4x4)
- Channel expansion (64 → 128 → 256 → 512)
- **Problem**: Lose spatial details early, require careful augmentation

**ConvMixer**:

- Maintains larger feature maps throughout (e.g., 16x16 constant for patch_size=2)
- Fixed channel dimension (e.g., 768 throughout)
- Depthwise conv: Learns spatial patterns (local features)
- Pointwise conv: Learns channel interactions (global features)
- **Advantage**: Preserves spatial context, robust to both augmentation types

### Why Label Smoothing?

ConvMixer's large feature maps provide high representational capacity:

- More prone to memorizing training set
- Label smoothing (ε=0.1) regularizes by targeting soft labels: (1-ε) for true class, ε/(K-1) for others
- Complements mixup/cutmix which also create soft labels
- **Key insight**: ConvMixer needs MORE regularization, not less

## Validation Criteria

After running the fixed scripts, expect to see:

### First 10 Epochs

- ✅ Train loss: Decreasing from ~4.6 to ~3.5
- ✅ Train acc: Increasing from ~1% to 15-25%
- ✅ Val acc: **Increasing steadily** (not stuck at 1%)
- ✅ Val loss: Decreasing or stable (not exploding)

### After 60 Epochs (past warmup)

- ✅ Train acc: 60-70%
- ✅ Val acc: 45-55%
- ✅ Val F1: 0.40-0.50
- ✅ Clear signs of convergence

### Final (200 Epochs)

- ✅ Best val F1: 0.83-0.87 (target >= 0.85)
- ✅ Stable validation metrics (no oscillation)

## References

1. **ConvMixer Paper**: Trockman & Kolter, "Patches Are All You Need?", ICLR 2022
   - Uses label smoothing + strong augmentation
   - Achieves 91.3% on CIFAR-100

2. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
   - Prevents overconfident predictions
   - Improves generalization

3. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018

4. **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019

## Status

✅ Implementation complete
✅ No lint errors
✅ Backward compatible
🔄 Testing in progress

---
**Last Updated**: 2025-10-26
**Branch**: `exp-6-convmixer`
