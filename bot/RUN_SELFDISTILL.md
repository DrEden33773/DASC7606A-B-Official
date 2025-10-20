# 🚀 立即运行自蒸馏实验

**模型**: Wide ResNet-28-10 + Self-Distillation (BYOT)  
**目标**: F1 ≥ 0.85  
**预期**: F1 = 0.84-0.86

---

## ⚡ 快速命令

### 最简化版本

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --use_self_distillation
```

**说明**: 其他参数使用最优默认值

---

### 完整版本

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 9 \
    --use_self_distillation \
    --distill_temperature 4.0 \
    --distill_alpha 0.9 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup_epochs 10 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --mixup_alpha 0.25 \
    --use_cutmix \
    --cutmix_alpha 0.65 \
    --use_amp \
    --use_ema \
    --seed 42
```

---

## 🎯 关键参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| model | wide_resnet28_10_selfdistill | 自蒸馏版本 |
| use_self_distillation | True | 启用自蒸馏 |
| distill_temperature | 4.0 | 知识蒸馏温度（论文建议）|
| distill_alpha | 0.9 | 软标签权重 90% |
| drop_path_rate | 0.1 | Phase 1 最佳值 |
| aug_strength | randaugment | Phase 1 最佳 |

---

## 📊 预期效果

```
Baseline (Phase 1): F1 = 0.8131
+ Self-Distillation (+3%): F1 = 0.8431
目标: F1 ≥ 0.85 ✅
```

**成功概率**: 70-75%

---

## 🔍 监控要点

### 健康指标

- Train Acc: 55-60% (正常，自蒸馏增加难度)
- Val Acc: 84-86%
- Best epoch: 250-300
- Val F1 持续上升

### 警报信号

- Train Acc < 50% → loss 权重问题
- Val F1 < 0.82 → 自蒸馏未生效
- Training 异常慢 → 可能实现有误

---

## 🎓 关键洞察

**模型容量甜点发现**:

```
36.5M (WRN-28-10) = 最优 ✅
52.8M (WRN-28-12) = 过拟合 ❌

结论: 不增大模型，用自蒸馏提升！
```

**自蒸馏优势**:

- 同一个模型内部知识转移
- 不需要额外训练 teacher
- 推理时可移除中间分类器

---

**准备就绪！立即开始训练！** 🚀

**预计时间**: 4-5 小时  
**查看详情**: `bot/experiments/phase2_5/exp_300_selfdistill.md`
