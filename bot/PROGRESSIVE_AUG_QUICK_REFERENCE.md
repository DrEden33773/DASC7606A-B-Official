# 🚀 渐进式增强快速参考

## 三种模式对比

| 模式 | 特点 | 适用场景 | 命令 |
|------|------|----------|------|
| **staged** ⭐推荐 | 三阶段阶梯式，清晰的转折点 | 默认首选，最稳定 | `--progressive_aug_mode staged` |
| **linear** | 平滑线性增长，无跳变 | 长时间训练（300+ epochs） | `--progressive_aug_mode linear` |
| **cosine** | 余弦增长，中期加速 | 实验性，可能有意外惊喜 | `--progressive_aug_mode cosine` |

---

## 快速命令

### 🥇 推荐命令（staged模式）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode staged
```

### 🥈 备选命令（linear模式）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode linear \
  --progressive_aug_peak_epoch 80
```

### 🥉 实验命令（cosine模式）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode cosine
```

---

## 组合优化

### Progressive Aug + SE-Net

```bash
python main.py --use_se --se_reduction 8 \
  --dropout 0.15 \
  --use_progressive_aug --progressive_aug_mode staged
```

### Progressive Aug + 延长Warmup

```bash
python main.py \
  --use_progressive_aug --progressive_aug_mode staged \
  --warmup_epochs 20
```

---

## 增强时间表（staged模式）

```
Epoch 1-30:   🟢 Weak   (RandAug m=5, Mixup=0.1, CutMix=0.3)
Epoch 31-100: 🟡 Medium (RandAug m=7, Mixup=0.2, CutMix=0.5)
Epoch 101+:   🔴 Strong (RandAug m=9, Mixup=0.25, CutMix=0.65)
```

---

## 预期改进

- ✅ 早停陷阱: 28 epochs → **10-15 epochs** (↓50%)
- ✅ 最终F1: 0.81 → **0.82-0.83** (+0.01-0.02)
- ✅ 训练稳定性: ⭐⭐⭐ → **⭐⭐⭐⭐⭐**

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_progressive_aug` | False | 启用渐进式增强 |
| `--progressive_aug_mode` | staged | 模式: staged/linear/cosine |
| `--progressive_aug_peak_epoch` | 100 | 达到峰值强度的epoch |
