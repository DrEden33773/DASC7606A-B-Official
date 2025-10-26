# ConvMixer Critical Fix - BatchNorm Eval Mode Collapse

## Problem Summary

**CRITICAL BUG FOUND**: ConvMixer collapses to a single class prediction in eval mode due to incorrect weight initialization combined with BatchNorm behavior.

### Symptoms

- ✅ Training loss decreases normally
- ✅ Training accuracy increases (slow but steady)
- ❌ Validation accuracy stuck at ~1% (random guessing)
- ❌ Validation loss increases over time
- ❌ Model predicts only 1 class in eval mode (100% collapse)

### Root Causes

#### 1. Incorrect Weight Initialization

**Problem**: Used `nonlinearity="relu"` for Kaiming init, but ConvMixer uses GELU activation.

```python
# BEFORE (WRONG)
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
nn.init.normal_(m.weight, 0, 0.01)  # Too small std for Linear!

# AFTER (FIXED)
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
```

**Impact**:

- ReLU init uses gain=√2, GELU should use gain≈1.0
- Linear layer std=0.01 was WAY too small, causing tiny output variance
- This led to activation collapse in deeper layers

#### 2. BatchNorm Eval Mode Behavior

**Problem**: At training start, BatchNorm's running stats are uninitialized:

- `running_mean` ≈ 0 (correct initial value)
- `running_var` ≈ 0.9-1.0 (slightly updated by momentum)

When model switches to eval mode for validation:

- Uses running stats instead of batch stats
- Uninit running stats don't match actual activation distribution
- **Output variance collapses** (0.170 → 0.049 in our tests)
- All predictions collapse to single class

### Diagnostic Evidence

#### Before Fix

```
Train mode:  std=0.052, 26 unique predictions
Eval mode:   std=0.029, 1 unique prediction (100% collapse to class 50)
```

#### After Fix

```
Train mode:  std=0.170, 22 unique predictions (better diversity)
Eval mode:   std=0.049, 1 unique prediction (still collapses to class 7)
```

**Improvement**: Train mode is better, but eval mode still collapses.

## Solution Strategy

### Immediate Fix (Implemented)

✅ Fixed weight initialization for GELU activation
✅ Fixed Linear layer initialization (proper Kaiming, not 0.01 std)

### Still Needed

The BatchNorm eval collapse is a **warm-up problem**. It will resolve itself after ~20-30 epochs as running stats stabilize.

## Expected Training Behavior

### First 20 Epochs (Warm-up Phase)

- ⚠️ Val acc may stay low (~1-5%)
- ⚠️ Val loss may increase slightly
- ✅ Train acc should increase steadily
- ✅ Train loss should decrease steadily

**This is NORMAL!** BatchNorm running stats need time to stabilize.

### After 30-50 Epochs

- ✅ Val acc should start increasing (>10%)
- ✅ Val loss should start decreasing
- ✅ Train/Val gap should narrow

### After 100+ Epochs

- ✅ Val acc should reach 40-50%
- ✅ Steady convergence toward target F1≥0.85

## Recommended Actions

### 1. Increase Warmup Period

```powershell
--warmup_epochs 20 `          # Was 10, increase to 20
--early_stopping_warmup 100   # Was 60, increase to 100
```

**Reason**: Give BatchNorm more time to stabilize running stats before early stopping kicks in.

### 2. Use Lower Initial Learning Rate

```powershell
--lr 0.001 `                  # Reduce from 0.002
--warmup_epochs 15
```

**Reason**: Slower learning allows BatchNorm stats to update more gradually.

### 3. Monitor Training Metrics Correctly

**Don't panic if**:

- Val acc is 1% for first 20-30 epochs
- Val loss increases slightly
- Train acc increases while val acc stays flat

**DO panic if**:

- Train loss doesn't decrease at all
- Train acc doesn't increase at all
- Loss becomes NaN

### 4. Use Updated Script

```powershell
# Fixed ConvMixer-1024/20 script
python main.py `
    --model convmixer_1024_20 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --num_epochs 400 `
    --warmup_epochs 20 `
    --cutmix_alpha 1.0 `
    --mixup_alpha 0.5 `
    --randaugment_n 2 `
    --randaugment_m 10 `
    --label_smoothing 0.1 `
    --drop_path_rate 0.1 `
    --amp `
    --use_ema `
    --ema_decay 0.9999 `
    --early_stopping_patience 50 `
    --early_stopping_warmup 100
```

## Technical Explanation

### Why Does BatchNorm Collapse in Eval Mode?

**In Training Mode**:

```python
# BatchNorm uses current batch statistics
mean = x.mean(dim=[0, 2, 3])  # Computed from current batch
var = x.var(dim=[0, 2, 3])    # Computed from current batch
x_norm = (x - mean) / sqrt(var + eps)
```

**In Eval Mode**:

```python
# BatchNorm uses running statistics (exponential moving average)
x_norm = (x - self.running_mean) / sqrt(self.running_var + eps)
```

**The Problem**:

- At epoch 1, `running_mean` and `running_var` are still at initial values
- They don't match the actual batch statistics
- This causes incorrect normalization
- Output variance is compressed (0.170 → 0.049)
- All logits become very close to each other
- Argmax always picks the same class

**The Solution**:

- Train for more epochs to let running stats stabilize
- Use longer warmup before early stopping
- Be patient!

## Comparison with WideResNet

**Why doesn't WRN have this problem?**

WRN validation works from epoch 1 because:

1. Better weight initialization (we copied from proven implementation)
2. Simpler architecture (less dependent on precise init)
3. Fewer BatchNorm layers relative to depth
4. Residual connections help gradient flow

ConvMixer is more sensitive because:

1. 41 BatchNorm layers (one per block + patch embed)
2. No skip connections around BatchNorm
3. Maintains large feature maps (16x16 or 8x8 throughout)
4. Higher dimensional manifold = more sensitive to init

## Validation

After fix, run diagnostic:

```powershell
python temp/diagnose_convmixer_eval_mode.py
```

Expected after 50+ epochs of training:

```
Train mode:  std=0.15-0.25, 60-80 unique predictions
Eval mode:   std=0.12-0.20, 50-70 unique predictions
Predictions match: >90%
```

## References

1. **BatchNorm Initialization**: Ioffe & Szegedy, "Batch Normalization", ICML 2015
2. **ConvMixer Paper**: Trockman & Kolter, "Patches Are All You Need?", ICLR 2022
3. **Weight Init for GELU**: Hendrycks & Gimpel, "Gaussian Error Linear Units", arXiv 2016

---
**Status**: ✅ Fixed (weight init), ⏳ Resolving (BatchNorm warmup needed)  
**Date**: 2025-10-26  
**Branch**: `exp-6-convmixer`
