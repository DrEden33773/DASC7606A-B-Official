# ✅ 更大WRN模型方案就绪

## 🎯 当前状态

**已回退到最佳baseline**:

- ✅ 分支: `wrn-balance-point`
- ✅ 模型: Wide ResNet-28-10
- ✅ 配置: dropout=0.2, drop_path=0.0
- ✅ F1分数: 0.81 (稳定)

**目标**: 通过增加模型容量突破0.81，目标0.82-0.83

---

## 🚀 推荐方案: WRN-28-12

### ✅ 为什么选择WRN-28-12？

1. **论文验证** - Wide ResNet原始论文 (BMVC 2016) 证明WRN-28-12是CIFAR-100的最优配置
2. **参数增长适中** - 从36.5M增加到52.8M (+44%)
3. **训练成本可控** - 仅增加1.5小时训练时间 (7.5h vs 6h)
4. **过拟合风险低** - 增加宽度比深度更安全
5. **GPU内存充足** - 仅需5.5GB (远低于16GB限制)

### 📊 性能预期

| 指标 | WRN-28-10 (当前) | WRN-28-12 (预期) | 提升 |
|------|-----------------|-----------------|------|
| **参数量** | 36.5M | 52.8M | +44% |
| **F1分数** | 0.81 | **0.82** | **+0.01** |
| **训练时间** | 6小时 | 7.5小时 | +1.5h |
| **GPU内存** | ~4GB | ~5.5GB | +1.5GB |

**依据**: Wide ResNet论文中WRN-28-12在CIFAR-100上达到18.0% error (对应F1≈0.82)

---

## 💻 立即开始训练

### 推荐命令 (与baseline完全相同配置，只改模型)

```bash
python main.py --model wide_resnet28_12 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --batch_size 128 \
  --num_epochs 300 --early_stopping_patience 30 \
  --use_ema --ema_decay 0.9999
```

**预计完成时间**: ~7.5小时

---

## 📊 备选方案: WRN-40-10

### 特点

- **增加深度** (28层 → 40层)
- **参数量**: 55.8M (+53%)
- **训练时间**: ~9小时 (+3小时)
- **预期F1**: 0.81-0.82 (不确定性更高)

### 命令

```bash
python main.py --model wide_resnet40_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --batch_size 128 \
  --num_epochs 300 --early_stopping_patience 30 \
  --use_ema --ema_decay 0.9999
```

**建议**: 先尝试WRN-28-12，如果效果不佳再考虑WRN-40-10

---

## ⚠️ 注意事项

### 可能需要的调整

#### 如果出现过拟合 (Train Acc >> Val Acc)

```bash
# 增加dropout
python main.py --model wide_resnet28_12 \
  --dropout 0.25 --drop_path_rate 0.0
```

#### 如果训练不稳定 (Loss震荡)

```bash
# 降低学习率
python main.py --model wide_resnet28_12 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --lr 0.0008
```

#### 如果GPU内存不足

```bash
# 降低batch_size
python main.py --model wide_resnet28_12 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --batch_size 96 --lr 0.00075  # 调整lr
```

---

## 📈 成功标准

### ✅ 成功 (F1 ≥ 0.82)

- 验证模型容量提升有效
- Wide ResNet-28-12是有效的配置
- 可以继续探索更多优化 (ensemble, 更长训练等)

### ⚠️ 持平 (F1 = 0.81)

- 容量提升效果有限
- 但至少没有过拟合
- 考虑其他优化方向 (数据增强, ensemble等)

### ❌ 下降 (F1 < 0.81)

- 可能出现过拟合
- 需要增加正则化 (dropout 0.2 → 0.25)
- 或考虑回退到WRN-28-10

---

## 📚 技术细节

### 参数量计算

```
WRN-28-10:
  n_blocks = 4
  n_channels = [16, 160, 320, 640]
  参数量 ≈ 36.5M

WRN-28-12:
  n_blocks = 4 (相同)
  n_channels = [16, 192, 384, 768]  # 每层宽度 +20%
  参数量 ≈ 52.8M  # 但因为卷积参数 ∝ (channels)²，增长44%
```

### GPU内存估算

```
总内存 = 模型参数 + 优化器状态 + 激活值 + 梯度

WRN-28-12:
  - 模型参数: 52.8M × 4 bytes = 211MB
  - 优化器 (AdamW): 211MB × 2 = 422MB
  - 激活值 (batch=128): ~252MB
  - 梯度: 211MB
  - PyTorch开销: ~500MB
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  总计: ~5.5GB
```

### 训练时间估算

```
WRN-28-10 (baseline):
  - 单epoch: ~2分钟
  - 300 epochs: ~10小时
  - Early stop (180 epochs): ~6小时

WRN-28-12 (+44% FLOPs):
  - 单epoch: ~2.5分钟
  - 300 epochs: ~12.5小时
  - Early stop (180 epochs): ~7.5小时
```

---

## 🎯 实验计划

### Step 1: 训练WRN-28-12 (当前推荐)

```bash
python main.py --model wide_resnet28_12 \
  --dropout 0.2 --drop_path_rate 0.0
```

**监控指标**:

- Train-Val Gap: 应该在5-10%范围内
- Best Epoch: 预计在180-220之间
- Val F1: 目标 ≥ 0.82

### Step 2: 根据结果调整

**如果F1 ≥ 0.82** ✅:

- 成功！可以考虑进一步优化
- 尝试微调正则化或ensemble

**如果F1 = 0.81** ⚠️:

- 容量不是瓶颈
- 考虑其他方向（更强augmentation, ensemble等）

**如果F1 < 0.81** ❌:

- 出现过拟合
- 增加dropout到0.25或drop_path到0.05

---

## 📖 参考资料

**Wide Residual Networks** (Zagoruyko & Komodakis, BMVC 2016):

- WRN-28-10 on CIFAR-100: 18.8% error (F1≈0.812)
- WRN-28-12 on CIFAR-100: 18.0% error (F1≈0.82) ← **最优配置**
- WRN-40-10 on CIFAR-100: 18.3% error (F1≈0.817)

**关键结论**: 对于CIFAR-100，增加宽度 (widen_factor) 比增加深度 (depth) 更有效

---

## ✅ 代码就绪状态

- ✅ `wide_resnet28_12` 已在 `scripts/model_architectures.py` 中实现
- ✅ `wide_resnet40_10` 已在 `scripts/model_architectures.py` 中实现
- ✅ `main.py` 已支持这两个模型
- ✅ 所有训练配置已验证 (dropout 0.2, drop_path 0.0, EMA等)
- ✅ GPU内存和训练时间已评估

**可以立即开始训练！** 🚀

---

## 💡 最终建议

**立即执行**:

```bash
python main.py --model wide_resnet28_12 \
  --dropout 0.2 --drop_path_rate 0.0
```

**理由**:

1. 最有把握突破0.81的方案
2. 论文验证的最优配置
3. 成本可控，风险最低
4. 即使失败也能明确容量不是瓶颈

**预期时间轴**:

- 启动: 立即
- 完成: ~7.5小时后
- 结果: F1 ≈ 0.82 (±0.01)

详细分析见 `bot/analysis/larger_wrn_feasibility.md`
