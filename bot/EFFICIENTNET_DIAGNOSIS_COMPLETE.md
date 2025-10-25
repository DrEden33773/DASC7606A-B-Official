# 🔍 EfficientNet 诊断完整报告

## 用户报告的问题

**现象**:

- WRN-28-12: Val Loss 和 Val Acc 在 warmup 期**缓慢变化** ✓
- EfficientNet-B1: Val Loss 和 Val Acc **始终不变，Val Acc 固定在 1%** ✗

## 根本原因分析

### 1. 激活消失问题 (已修复 ✅)

**原因**:

- EfficientNet 的 SEBlock 中 Linear 层使用了 `bias=False`
- Linear 层的初始化 `std=0.01` 过小

**后果**:

- SEBlock excitation 输出极小 (Std: 0.000003)
- 导致特征被过度抑制，激活值逐层衰减至 0
- 最终 head 输出全为 0，模型无法学习

**修复**:

1. ✅ SEBlock Linear 层改为 `bias=True`
2. ✅ Linear 层初始化改为 `xavier_uniform_()`

### 2. EMA + torch.compile 冲突 (已修复 ✅)

**原因**:

- `torch.compile()` 创建 wrapper，阻止 EMA 正确更新参数

**后果**:

- EMA shadow weights 从未更新
- 验证时使用初始随机权重
- Val metrics 冻结不变

**修复**:

1. ✅ `build_model()` 返回 `(compiled_model, original_model)`
2. ✅ EMA 使用 `original_model` 初始化和更新
3. ✅ 验证时使用 `original_model`

### 3. 训练缓慢问题 (部分修复 ⚠️)

**现状**:

- ✅ 激活值正常 (Std: 0.64)
- ✅ 梯度流正常 (平均梯度范数: 6.9)
- ⚠️ 但 Loss 下降很慢，Acc 增长缓慢

**可能原因**:

1. **初始化策略**: EfficientNet 可能需要更精细的初始化
2. **学习率**: SGD lr=0.1 可能对 EfficientNet 过大
3. **Batch Size**: 小 batch (32) 可能导致训练不稳定
4. **Optimizer**: EfficientNet 可能更适合 AdamW/RMSProp

## 代码修复总结

### `scripts/model_architectures.py`

#### 1. SEBlock Linear 层添加 bias

```python
# 修复前
self.excitation = nn.Sequential(
    nn.Linear(channels, reduced_channels, bias=False),  # ❌
    Swish(),
    nn.Linear(reduced_channels, channels, bias=False),   # ❌
    nn.Sigmoid(),
)

# 修复后
self.excitation = nn.Sequential(
    nn.Linear(channels, reduced_channels, bias=True),   # ✅
    Swish(),
    nn.Linear(reduced_channels, channels, bias=True),    # ✅
    nn.Sigmoid(),
)
```

#### 2. Linear 层初始化改进

```python
# 修复前
elif isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, 0, 0.01)  # ❌ std=0.01 太小
    nn.init.constant_(m.bias, 0)

# 修复后
elif isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)   # ✅ 使用 Xavier 初始化
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
```

### `main.py` 和 `scripts/train_utils.py`

EMA + torch.compile 修复 (见 `bot/CRITICAL_VAL_LOSS_BUG_FIX.md`)

## EfficientNet vs WRN 差异分析

| 特性 | WideResNet | EfficientNet | 影响 |
|------|-----------|--------------|------|
| **激活函数** | ReLU | Swish | Swish 在负值区域梯度更小 |
| **架构** | ResNet blocks | MBConv + SE | SE 可能过度抑制特征 |
| **初始化** | Kaiming (ReLU) | Kaiming (ReLU) | 不匹配 Swish |
| **参数量** | 52.59M | 6.64M | EfficientNet 更小，容量更低 |
| **训练稳定性** | 高 (成熟架构) | 中 (需要调参) | EfficientNet 对超参数更敏感 |

## 优化器兼容性分析

### SGD (当前测试)

**优点**:

- 对 WRN 效果好
- 稳定性高

**缺点**:

- EfficientNet 训练慢
- 需要精细的学习率调整
- 对初始化敏感

### AdamW (推荐 ✅)

**优点**:

- **自适应学习率**，对不同层自动调整
- **对初始化不敏感**
- EfficientNet 论文使用 RMSProp (类似)
- 训练更稳定

**缺点**:

- 可能需要更多 epochs
- Weight decay 需要调整

### 推荐配置

```powershell
# EfficientNet-B1 推荐配置
python main.py `
    --model efficientnet_b1 `
    --input_size 64 `
    --batch_size 128 `
    --num_epochs 300 `
    --optimizer adamw `      # ✅ 推荐 AdamW
    --lr 0.001 `            # AdamW 典型学习率
    --weight_decay 0.01 `   # AdamW 推荐值
    --warmup_epochs 10 `
    --scheduler cosine `
    --use_ema `
    --ema_decay 0.9999 `
    --use_compile
```

## 最终建议

### 1. 立即测试 (10 epochs)

```powershell
# 测试修复后的 EfficientNet (使用 AdamW)
python main.py `
    --model efficientnet_b1 `
    --input_size 64 `
    --batch_size 192 `
    --num_epochs 10 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --warmup_epochs 5 `
    --use_ema `
    --use_compile
```

**预期结果**:

- ✅ Val Loss 应该开始下降 (不再冻结)
- ✅ Val Acc 应该 > 5% (不再固定 1%)
- ✅ Val F1 应该 > 0.05 (不再固定 0.0002)

### 2. 如果还有问题

**Plan B**: 尝试不使用 SE Block

```python
# 在 MBConvBlock 初始化中临时禁用 SE
# se_ratio=0.25 → se_ratio=0  # 完全禁用 SE
```

**Plan C**: 使用更简单的 EfficientNet-B0

```powershell
--model efficientnet_b0  # 更小，更容易训练
```

## 状态总结

| 问题 | 状态 | 说明 |
|------|------|------|
| 激活消失 | ✅ 已修复 | Linear层 bias + Xavier初始化 |
| EMA冻结 | ✅ 已修复 | 分离 compiled 和 original model |
| Val metrics不变 | ✅ 应已修复 | 两个bug都修了 |
| 训练缓慢 | ⚠️ 需要验证 | 建议使用 AdamW |
| SGD兼容性 | ⚠️ 待确认 | 可能不是最佳选择 |

## 下一步

1. ✅ 代码已全部修复
2. ⏳ 运行 10 epoch 测试验证修复效果
3. ⏳ 根据结果决定是否需要调整优化器
4. ⏳ 如果成功，运行完整训练 (300 epochs)

---

**修复完成时间**: 2025-10-25
**修复文件**:

- `scripts/model_architectures.py` (SEBlock + 初始化)
- `main.py` (EMA + torch.compile)
- `scripts/train_utils.py` (EMA update)
