# 🚀 Long-Board Loss Weighting 快速启动

## 一键启动

```bash
# 基本配置（推荐）
python main.py --model wide_resnet28_12 \
  --use_class_weights \
  --weight_strategy long_board \
  --dropout 0.2 \
  --drop_path_rate 0.0 \
  --num_epochs 400 \
  --early_stopping_patience 50
```

**预期**: F1 = 0.84-0.85 (+0.02-0.03 vs baseline 0.82)  
**训练时间**: ~2h30m  
**成功率**: 高

---

## 🎯 核心原理（1分钟理解）

### 传统思维 ❌

```
试图提升 boy/girl/man/woman 等低分类 (F1=0.57)
→ 受限于32×32分辨率，多次失败
→ 浪费训练资源
```

### Long-Board策略 ✅

```
降低 已很高类别的关注 (F1≥0.93): weight=0.75
大幅提高 中等类别的关注 (F1 0.70-0.79): weight=1.6
保持 detail-sensitive 正常 (F1<0.70): weight=1.0

→ 聚焦有改进潜力的82个类别
→ 预期整体F1提升0.02-0.03
```

---

## 📊 权重分配一览

| 类别组 | F1范围 | 权重 | 数量 | 策略 |
|--------|--------|------|------|------|
| 极高分 | ≥0.93 | 0.75 | 6个 | 降低关注 |
| 高分 | 0.88-0.92 | 0.9 | 13个 | 轻微降低 |
| 中高分 | 0.80-0.87 | 1.1 | 45个 | 轻微提高 |
| **中等** | **0.70-0.79** | **1.6** | **18个** | **大幅提高** ← 关键 |
| Detail-sensitive | <0.70 | 1.0 | 9个 | 保持正常 |
| 其他低分 | <0.70 | 1.4 | 9个 | 适度提高 |

---

## 🔧 快速调试

### 检查是否正确启用

启动训练后，应该看到：

```
✅ Using LONG-BOARD class weighting strategy:
   • Extreme high-score (F1≥0.93): weight=0.75 (reduce attention)
   • Medium score (F1 0.70-0.79): weight=1.6 (major increase)
   Strategy: Focus on classes with most improvement potential!
   Using WeightedLossWrapper for optimal long-board effect with mixup/cutmix
```

### 常见问题

**Q: Label smoothing被禁用了？**  
A: 正常。Long-board与label smoothing不兼容。

**Q: 需要修改其他参数吗？**  
A: 不需要。Long-board与现有配置完全兼容。

**Q: 可以与Ensemble组合吗？**  
A: 强烈推荐！预期3模型Ensemble可达0.85-0.86。

---

## 📈 预期提升

```
Baseline:   0.82
Long-Board: 0.84-0.85 (+0.02-0.03)
+ Ensemble: 0.85-0.86 (+0.03-0.04)
```

---

## 💡 为什么有效？

```
Macro F1 = (Σ所有类别F1) / 100

提升中等类别(18个) 0.73→0.78 (+0.05)
→ 贡献 +0.009 到整体F1

提升中高分类别(45个) 0.82→0.86 (+0.04)
→ 贡献 +0.018 到整体F1

总计: +0.027 → 实际约+0.02-0.03
```

---

## 🎉 立即开始

```bash
python main.py --model wide_resnet28_12 \
  --use_class_weights \
  --weight_strategy long_board \
  --num_epochs 400
```

详细文档: `bot/LONG_BOARD_LOSS_WEIGHTING_GUIDE.md`
