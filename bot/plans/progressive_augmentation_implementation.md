# 渐进式数据增强（Progressive Augmentation）实现方案

## 📋 概述

**目标**：解决训练初期的augmentation-regularization不平衡问题，通过动态调整数据增强强度来提升训练稳定性和最终性能。

**核心思想**：

- **训练初期（Warmup + Early Stage）**：使用较弱的augmentation，让模型快速学习基本特征
- **训练中期（Middle Stage）**：逐步增强augmentation，平衡过拟合风险
- **训练后期（Late Stage）**：使用最强augmentation，最大化泛化能力

---

## 🎯 问题分析回顾

### 当前问题

```
Epoch 1-32: 早停陷阱
  Train Loss: 4.13 → 2.86 (下降，模型在记忆)
  Val Loss:   4.62 → 6.03 (上升，无法泛化)
  Val F1:     0.0032 (停滞不前)
  
原因: RandAugment(n=2,m=9) + Mixup(0.25) + CutMix(0.65)
      对于随机初始化的模型太强，导致梯度信号混乱
```

### 预期改进

```
渐进式Augmentation后:
  Epoch 1-10:   弱增强 → Val Loss稳定下降
  Epoch 11-50:  中等增强 → 模型开始泛化
  Epoch 51+:    强增强 → 最大化泛化能力
  
结果: 减少早停陷阱持续时间，提升最终F1 (预期 0.81 → 0.82)
```

---

## 🛠️ 实现方案

### 方案A：三阶段阶梯式增强（推荐）

#### 增强策略表

| Stage | Epochs | RandAugment (n, m) | Mixup α | CutMix α | 说明 |
|-------|--------|-------------------|---------|----------|------|
| **Early** | 1-30 | (1, 5) | 0.1 | 0.3 | 让模型先学会基本分类 |
| **Middle** | 31-100 | (2, 7) | 0.2 | 0.5 | 逐步增强正则化 |
| **Late** | 101+ | (2, 9) | 0.25 | 0.65 | 最大化泛化（当前默认值） |

#### 实现位置

- **文件**: `scripts/train_utils.py`
- **修改点**: `train_epoch()` 函数
- **逻辑**: 根据当前epoch动态调整augmentation参数

---

### 方案B：线性渐进增强（更平滑）

#### 数学公式

```python
def get_progressive_aug_params(current_epoch, total_epochs=300):
    # RandAugment m: 5 → 9 线性增长
    m_start, m_end = 5, 9
    m = m_start + (m_end - m_start) * min(current_epoch / 100, 1.0)
    
    # Mixup alpha: 0.1 → 0.25 线性增长
    mixup_start, mixup_end = 0.1, 0.25
    mixup_alpha = mixup_start + (mixup_end - mixup_start) * min(current_epoch / 100, 1.0)
    
    # CutMix alpha: 0.3 → 0.65 线性增长
    cutmix_start, cutmix_end = 0.3, 0.65
    cutmix_alpha = cutmix_start + (cutmix_end - cutmix_start) * min(current_epoch / 100, 1.0)
    
    return int(m), mixup_alpha, cutmix_alpha
```

#### 可视化

```
RandAugment m:
9 |                    ══════════════════
  |                ／
7 |            ／
  |        ／
5 |════／
  +────────────────────────────────────
    0   20   40   60   80  100  120  epochs
```

---

### 方案C：余弦退火式增强（最激进）

#### 逻辑

类似Cosine Annealing LR，但反向操作（从弱到强）

```python
import math

def get_cosine_progressive_aug_params(current_epoch, warmup_epochs=30, peak_epoch=100):
    if current_epoch < warmup_epochs:
        # Warmup阶段：保持最弱
        return 5, 0.1, 0.3
    
    # 余弦增长
    progress = min((current_epoch - warmup_epochs) / (peak_epoch - warmup_epochs), 1.0)
    cosine_progress = (1 - math.cos(progress * math.pi)) / 2  # 0 → 1
    
    m = 5 + int(4 * cosine_progress)  # 5 → 9
    mixup_alpha = 0.1 + 0.15 * cosine_progress  # 0.1 → 0.25
    cutmix_alpha = 0.3 + 0.35 * cosine_progress  # 0.3 → 0.65
    
    return m, mixup_alpha, cutmix_alpha
```

---

## 📝 代码实现

### Step 1: 修改 `main.py` - 添加参数

```python
# 在 parse_args() 中添加
parser.add_argument(
    "--use_progressive_aug",
    action="store_true",
    help="Enable progressive augmentation (gradually increase augmentation strength during training)"
)
parser.add_argument(
    "--progressive_aug_mode",
    type=str,
    choices=["staged", "linear", "cosine"],
    default="staged",
    help="Progressive augmentation mode: staged (3-stage), linear, or cosine"
)
parser.add_argument(
    "--progressive_aug_peak_epoch",
    type=int,
    default=100,
    help="Epoch at which augmentation reaches maximum strength (default: 100)"
)
```

### Step 2: 修改 `scripts/train_utils.py` - 核心实现

#### 2.1 添加渐进式增强参数计算函数

```python
def get_progressive_augmentation_params(
    current_epoch: int,
    mode: str = "staged",
    peak_epoch: int = 100,
) -> Tuple[int, int, float, float]:
    """
    计算当前epoch的渐进式augmentation参数
    
    Args:
        current_epoch: 当前训练epoch (1-based)
        mode: 渐进模式 ("staged", "linear", "cosine")
        peak_epoch: augmentation达到最大强度的epoch
    
    Returns:
        (randaugment_n, randaugment_m, mixup_alpha, cutmix_alpha)
    """
    if mode == "staged":
        # 三阶段阶梯式
        if current_epoch <= 30:
            # Early Stage
            return 1, 5, 0.1, 0.3
        elif current_epoch <= 100:
            # Middle Stage
            return 2, 7, 0.2, 0.5
        else:
            # Late Stage
            return 2, 9, 0.25, 0.65
    
    elif mode == "linear":
        # 线性增长
        progress = min(current_epoch / peak_epoch, 1.0)
        
        # RandAugment: n保持2, m从5线性增长到9
        n = 2
        m = 5 + int(4 * progress)
        
        # Mixup: 0.1 → 0.25
        mixup_alpha = 0.1 + 0.15 * progress
        
        # CutMix: 0.3 → 0.65
        cutmix_alpha = 0.3 + 0.35 * progress
        
        return n, m, mixup_alpha, cutmix_alpha
    
    elif mode == "cosine":
        # 余弦增长
        import math
        
        warmup_epochs = 30
        if current_epoch < warmup_epochs:
            return 1, 5, 0.1, 0.3
        
        progress = min((current_epoch - warmup_epochs) / (peak_epoch - warmup_epochs), 1.0)
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2
        
        n = 2
        m = 5 + int(4 * cosine_progress)
        mixup_alpha = 0.1 + 0.15 * cosine_progress
        cutmix_alpha = 0.3 + 0.35 * cosine_progress
        
        return n, m, mixup_alpha, cutmix_alpha
    
    else:
        raise ValueError(f"Unknown progressive augmentation mode: {mode}")
```

#### 2.2 修改 `train_epoch()` 函数签名

```python
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    ema: Optional[ModelEMA] = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    use_cutmix: bool = False,
    # 新增参数
    use_progressive_aug: bool = False,
    progressive_aug_mode: str = "staged",
    progressive_aug_peak_epoch: int = 100,
    current_epoch: int = 1,
) -> Tuple[float, float]:
    """
    Train for one epoch with optional progressive augmentation.
    
    New Args:
        use_progressive_aug: Whether to use progressive augmentation
        progressive_aug_mode: Mode for progressive augmentation
        progressive_aug_peak_epoch: Epoch at which augmentation reaches peak
        current_epoch: Current training epoch (for progressive augmentation)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 如果启用渐进式增强，覆盖传入的augmentation参数
    if use_progressive_aug:
        _, randaug_m, mixup_alpha, cutmix_alpha = get_progressive_augmentation_params(
            current_epoch=current_epoch,
            mode=progressive_aug_mode,
            peak_epoch=progressive_aug_peak_epoch,
        )
        # 注意: RandAugment的n参数需要在DataLoader创建时设置，这里只能调整m
        # 实际上我们需要在每个epoch重新创建DataLoader（见下文）
    
    # ... 其余训练逻辑保持不变
```

#### 2.3 重要：需要在每个epoch动态创建DataLoader

**问题**: RandAugment的参数在Transform创建时就固定了，无法在训练中途修改。

**解决方案**: 如果启用`use_progressive_aug`，需要在每个epoch重新创建DataLoader。

```python
# 在 main.py 的 train() 函数中修改

def train(args, model):
    # ... 初始化部分保持不变 ...
    
    # 如果使用渐进式增强，不在这里创建DataLoader
    if not args.use_progressive_aug:
        train_loader, val_loader = load_data(...)
    else:
        val_loader = load_data(..., only_val=True)  # 验证集只创建一次
    
    for epoch in range(1, args.num_epochs + 1):
        # 如果启用渐进式增强，每个epoch动态创建train_loader
        if args.use_progressive_aug:
            n, m, mixup_alpha, cutmix_alpha = get_progressive_augmentation_params(
                current_epoch=epoch,
                mode=args.progressive_aug_mode,
                peak_epoch=args.progressive_aug_peak_epoch,
            )
            
            logger.info(
                f"[Progressive Aug] Epoch {epoch}: RandAugment(n={n}, m={m}), "
                f"Mixup α={mixup_alpha:.2f}, CutMix α={cutmix_alpha:.2f}"
            )
            
            train_loader, _ = load_data(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                randaugment_n=n,
                randaugment_m=m,
                # ... 其他参数
            )
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            # ...
            mixup_alpha=mixup_alpha if args.use_progressive_aug else args.mixup_alpha,
            cutmix_alpha=cutmix_alpha if args.use_progressive_aug else args.cutmix_alpha,
            current_epoch=epoch,
        )
        
        # ... 验证和保存逻辑
```

---

## ⚠️ 实现注意事项

### 1. DataLoader重建的性能开销

每个epoch重新创建DataLoader会有轻微的性能开销（~1-2秒），但相比训练时间（每epoch 1-2分钟）可以忽略。

### 2. RandAugment的n参数

- **n**: 应用的augmentation操作数量（建议保持1-2，变化不大）
- **m**: augmentation的强度（这是主要调整的参数）

### 3. Early Stopping的影响

渐进式augmentation可能导致后期性能波动，建议：

```python
# 在后期（如epoch > 100）适当放宽early stopping tolerance
if epoch > 100:
    effective_patience = args.early_stopping_patience * 1.5
```

### 4. 验证集不使用augmentation

确保`val_loader`始终使用干净的数据（无RandAugment/Mixup/CutMix）。

---

## 📊 实验计划

### 实验1：三阶段阶梯式（最稳定）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode staged \
  --progressive_aug_peak_epoch 100
```

**预期**:

- 减少早停陷阱持续时间（32 epochs → 15 epochs）
- 最终F1提升 0.01-0.02 (0.81 → 0.82-0.83)

### 实验2：线性渐进（最平滑）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode linear \
  --progressive_aug_peak_epoch 80
```

**预期**:

- 训练曲线更平滑
- 可能需要更多epochs才能收敛

### 实验3：余弦增长（最激进）

```bash
python main.py --model wide_resnet28_10 \
  --dropout 0.2 --drop_path_rate 0.0 \
  --use_progressive_aug --progressive_aug_mode cosine \
  --progressive_aug_peak_epoch 100
```

**预期**:

- 初期增长缓慢，中期加速
- 可能在50-80 epochs区间有性能突破

---

## 🔬 与其他优化的组合

### 组合1: Progressive Aug + SE-Net

```bash
python main.py --use_se --se_reduction 8 \
  --dropout 0.15 \  # 降低dropout，因为augmentation初期较弱
  --use_progressive_aug --progressive_aug_mode staged
```

**逻辑**: 渐进式augmentation解决训练初期不稳定，SE-Net提升特征表达能力。

### 组合2: Progressive Aug + Extended Warmup

```bash
python main.py \
  --use_progressive_aug --progressive_aug_mode staged \
  --warmup_epochs 20 \  # 延长warmup
  --progressive_aug_peak_epoch 120  # 更晚达到峰值
```

**逻辑**: 双重保护，最大化训练稳定性。

---

## 📈 预期效果

### 训练曲线对比

```
Without Progressive Aug:
Val Loss:
6.0 |     ╱╲╱╲╱╲    ╱╲
    |    ╱        ╲╱    ╲╱╲___
4.6 |═══╱                      ╲___
    |                                ╲___
1.0 |                                    ╲___
    +──────────────────────────────────────────
      0  10  20  30  40  50  ... 200  epochs

With Progressive Aug (Staged):
Val Loss:
4.6 |═══╲
    |     ╲___
    |          ╲___
    |               ╲___
1.0 |                    ╲_______________
    +──────────────────────────────────────────
      0  10  20  30  40  50  ... 200  epochs
```

### 性能提升预估

| 指标 | Baseline | Progressive Aug | 提升 |
|------|----------|-----------------|------|
| 早停陷阱持续时间 | 28 epochs | **10-15 epochs** | ↓ 50% |
| 最终验证F1 | 0.81 | **0.82-0.83** | +0.01-0.02 |
| 训练稳定性 | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** | 显著提升 |
| Detail-Sensitive Classes F1 | 0.56-0.60 | **0.58-0.63** | +0.02-0.03 |

---

## 🎯 推荐策略

1. **首选**: 三阶段阶梯式（`staged`）
   - 最稳定，易于调试
   - 清晰的阶段划分便于分析

2. **次选**: 线性渐进（`linear`）
   - 如果阶梯式仍有波动
   - 更适合长时间训练（300+ epochs）

3. **实验性**: 余弦增长（`cosine`）
   - 如果前两者都不理想
   - 可能带来意外惊喜

---

## 🔧 快速开始

### 最小改动实现（仅针对关键参数）

如果担心完整实现太复杂，可以先实现一个**简化版本**：

```python
# 在 main.py 的训练循环中添加
for epoch in range(1, args.num_epochs + 1):
    # 简单的两阶段augmentation
    if epoch <= 30:
        # Early stage: 弱增强
        current_mixup_alpha = 0.1
        current_cutmix_alpha = 0.3
    else:
        # Late stage: 正常强度
        current_mixup_alpha = args.mixup_alpha
        current_cutmix_alpha = args.cutmix_alpha
    
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer,
        device=args.device,
        mixup_alpha=current_mixup_alpha,
        cutmix_alpha=current_cutmix_alpha,
        # ... 其他参数
    )
```

**优点**: 无需修改DataLoader，只调整Mixup/CutMix强度
**缺点**: 无法调整RandAugment强度

---

## 📚 参考文献

1. **AutoAugment** (Cubuk et al., 2019): 自适应数据增强策略搜索
2. **Curriculum Learning** (Bengio et al., 2009): 从简单到困难的学习策略
3. **Progressive Growing of GANs** (Karras et al., 2018): 渐进式训练思想

---

## ✅ 总结

渐进式数据增强是解决当前**早停陷阱**问题的最直接、最有效的方法。通过在训练初期降低augmentation强度，让模型先学会基本特征，再逐步增强正则化，可以显著提升训练稳定性和最终性能。

**下一步行动**:

1. 实现三阶段阶梯式Progressive Aug
2. 运行实验验证效果
3. 如有必要，结合SE-Net或其他优化策略

预期最终F1可达 **0.82-0.83**，解决Detail-Sensitive Classes表现不佳的问题。
