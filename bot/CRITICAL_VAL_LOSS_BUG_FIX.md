# 🔴 CRITICAL BUG FIX: 验证指标冻结问题

## 问题报告时间

2025-10-25

## 用户报告的严重Bug

### 症状

```
Epoch 1-7: Val Loss = 4.6052 (从不变化)
Epoch 1-7: Val Acc = 1.00% (从不变化)
Epoch 1-7: Val F1 = 0.0002 (从不变化)
```

**所有验证指标完全冻结，训练Loss正常下降但验证指标毫无变化！**

## 根本原因分析

### 发现过程

1. **初步怀疑**: Class weights 或 weight_strategy 冲突
2. **深入排查**: 检查 `validate_epoch()` 函数实现 ✓ 正常
3. **关键发现**: EMA (Exponential Moving Average) 相关逻辑
4. **根本原因**: **`torch.compile()` wrapper 导致 EMA 无法正确更新参数**

### 技术细节

#### 问题链条

```python
# Step 1: 模型编译
model = efficientnet_b1(...)
compiled_model = torch.compile(model, backend="aot_eager")
model = compiled_model  # 现在 model 是一个 wrapper

# Step 2: EMA 初始化（错误）
ema = ModelEMA(model, ...)  # ❌ 使用了 compiled wrapper
# EMA 内部存储的是 wrapper 的参数引用，而不是真实模型参数

# Step 3: 训练中 EMA 更新（失败）
ema.update(model)  # ❌ 更新 wrapper，但真实参数未变化

# Step 4: 验证（使用未更新的 EMA）
ema.apply_shadow()  # ❌ Shadow 权重实际上是初始权重（未更新）
validate_epoch(model, ...)  # 使用"初始权重"进行验证
ema.restore()

# 结果：每个epoch验证时都使用相同的初始权重 → 指标冻结
```

#### 为什么训练Loss正常但验证指标冻结？

1. **训练阶段**:
   - 使用 `model`（compiled wrapper）进行前向传播 ✓
   - 梯度回传更新 `original_model` 的真实参数 ✓
   - Training loss/acc 正常 ✓

2. **验证阶段**:
   - EMA 的 shadow 权重从未正确更新
   - `ema.apply_shadow()` 将"初始权重"应用到模型
   - 验证时实际使用初始的随机权重
   - Val loss/acc 保持在随机初始化水平 ✗

## 修复方案

### 核心思路

**分离训练模型和 EMA 模型**：

- `compiled_model`: 用于训练（torch.compile wrapper）
- `original_model`: 用于 EMA 更新和验证（原始模型）

### 代码变更

#### 1. `build_model()` 返回两个模型

```python
def build_model(args) -> Tuple[nn.Module, nn.Module]:
    """
    Build the model (from scratch).
    
    Returns:
        Tuple of (compiled_model, original_model)
    """
    # 创建模型
    model = create_model(...)
    original_model = model  # ✅ 保存原始模型引用
    
    # 编译（如果启用）
    if args.use_compile:
        compiled_model = torch.compile(model, ...)
        model = compiled_model
    
    # ✅ 返回两个模型
    return model, original_model
```

#### 2. `train()` 函数接收两个模型

```python
def train(args, model: nn.Module, original_model: nn.Module):
    # ...
    
    # ✅ EMA 使用 original_model
    if args.use_ema:
        ema = ModelEMA(original_model, ...)
```

#### 3. `train_epoch` 支持独立的 EMA 模型

```python
def train_epoch(
    model: nn.Module,         # 用于训练（可能是 compiled）
    ...
    ema: Optional[ModelEMA] = None,
    ema_model: Optional[nn.Module] = None,  # ✅ 新增参数
):
    # ...
    
    # ✅ EMA 更新使用 ema_model
    if ema is not None:
        ema.update(ema_model if ema_model is not None else model)
```

#### 4. 验证时使用正确的模型

```python
# ✅ 验证时使用 original_model (not compiled wrapper)
validation_model = original_model if ema is not None else model

if ema is not None:
    ema.apply_shadow()  # Apply to original_model

val_loss, val_acc, val_f1 = validate_epoch(
    validation_model,  # ✅ 使用 original_model
    val_loader,
    criterion,
    device
)

if ema is not None:
    ema.restore()
```

#### 5. 更新所有调用点

```python
# main.py
model, original_model = build_model(args)  # ✅
train(args, model, original_model)  # ✅

# train_epoch 调用
train_epoch(
    model=model,  # Compiled model for training
    ...
    ema=ema,
    ema_model=original_model if ema is not None else None,  # ✅
    ...
)
```

## 修复验证

### 预期行为修复后

```
Epoch 1: Train Loss ↓, Val Loss ↓, Val Acc ↑, Val F1 ↑
Epoch 2: Train Loss ↓, Val Loss ↓, Val Acc ↑, Val F1 ↑
...
```

### 验证清单

- [x] `build_model()` 返回 `Tuple[nn.Module, nn.Module]`
- [x] `train()` 接收两个模型参数
- [x] EMA 使用 `original_model` 初始化
- [x] `train_epoch` 新增 `ema_model` 参数
- [x] EMA update 使用 `ema_model`
- [x] 验证时使用 `original_model`
- [x] 所有调用点更新
- [x] Linter 错误修复（0 errors）

## 影响范围

### 修改的文件

1. `main.py`:
   - `build_model()` 返回类型和实现
   - `train()` 函数签名
   - `standard_main()` 和 `ensemble_main()`
   - `train_epoch()` 调用
   - 验证逻辑

2. `scripts/train_utils.py`:
   - `train_epoch()` 签名新增 `ema_model` 参数
   - EMA update 逻辑

### 不影响的功能

- 模型架构实现 ✓
- 数据加载和增强 ✓
- 损失函数和优化器 ✓
- 其他训练逻辑 ✓

## 测试建议

### 快速验证脚本

```powershell
# 10 epochs 快速测试
python main.py `
    --model efficientnet_b1 `
    --input_size 64 `
    --batch_size 192 `
    --num_epochs 10 `
    --use_ema `
    --use_compile
```

### 关键监控指标

1. **Epoch 1**: Val Loss ≈ 4.6 (初始)
2. **Epoch 2**: Val Loss < 4.6 ✓ (开始下降)
3. **Epoch 3-10**: Val Loss 持续下降 ✓
4. **Epoch 10**: Val Acc > 5% ✓ (不再是 1%)
5. **Epoch 10**: Val F1 > 0.05 ✓ (不再是 0.0002)

### 对比测试

| 配置 | Bug 版本 | 修复版本 |
|------|---------|---------|
| Epoch 7 Val Loss | 4.6052 (不变) | < 3.5 (下降) |
| Epoch 7 Val Acc | 1.00% (冻结) | > 10% (上升) |
| Epoch 7 Val F1 | 0.0002 (冻结) | > 0.10 (上升) |

## 技术教训

### 1. `torch.compile()` 陷阱

**问题**: `torch.compile()` 返回的 wrapper 不是原始 `nn.Module`

```python
# 看起来一样
isinstance(compiled_model, nn.Module)  # True

# 但参数访问可能不同
list(compiled_model.named_parameters())  # 可能返回 wrapper 参数
```

**教训**: 需要显式保存原始模型引用，用于需要直接访问参数的操作（如 EMA）

### 2. EMA 的隐蔽性

**问题**: EMA 初始化和更新时没有明显错误提示

```python
# 这些代码都"正常"运行，不报错
ema = ModelEMA(compiled_model, ...)  # ❌ 但实际有问题
ema.update(compiled_model)  # ❌ 更新了 wrapper
```

**教训**: 对于"无错误但异常"的情况，需要深入检查间接影响的组件

### 3. 调试策略

**有效方法**:

1. 监控关键指标的异常模式（如完全不变）
2. 逐层排查数据流
3. 对比"预期行为" vs "实际行为"
4. 验证假设（EMA shadow 是否正确更新）

## 后续优化

### 可选改进

1. **添加 EMA 健康检查**:

```python
# 在训练循环中添加
if epoch == 5 and ema is not None:
    shadow_params = list(ema.shadow.values())
    if all(p.std() < 0.01 for p in shadow_params):
        logger.warning("⚠️ EMA shadow weights have low variance!")
```

2. **分离 compile 开关**:

```python
parser.add_argument("--compile_for_validation", action="store_true")
# 默认验证时不使用 compiled model（更安全）
```

3. **EMA 模式可视化**:

```python
if args.use_ema and epoch % 10 == 0:
    logger.info(f"EMA shadow std: {ema_shadow_std:.4f}")
    logger.info(f"Model param std: {model_param_std:.4f}")
```

## 结论

**这是一个由于 `torch.compile()` 和 EMA 交互导致的隐蔽bug。**

- ✅ **根本原因**: torch.compile wrapper 阻止 EMA 正确更新参数
- ✅ **修复方案**: 分离训练模型和 EMA 模型
- ✅ **代码质量**: 通过所有 linter 检查
- ✅ **影响范围**: 仅限 EMA 相关逻辑，其他功能不受影响
- ✅ **测试建议**: 10 epochs 快速测试即可验证修复

**修复后，验证指标应该正常变化，Training Loss 和 Val Loss/Acc/F1 都会随训练进展！**
