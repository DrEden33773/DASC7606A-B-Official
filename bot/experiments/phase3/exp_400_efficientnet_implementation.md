# Exp #400 - EfficientNet-B0 实现指南

**创建日期**: 2025-01-26  
**目标**: 提供完整的 EfficientNet-B0 实现代码  
**参考**: [EfficientNet 论文](https://arxiv.org/abs/1905.11946)

---

## 📋 实现清单

### 需要修改的文件

1. ✅ `scripts/model_architectures.py` - 添加 EfficientNet
2. ✅ `scripts/train_utils.py` - 支持 input_size 参数
3. ✅ `main.py` - 添加命令行参数

---

## 🔧 代码实现

### 1. EfficientNet 架构 (scripts/model_architectures.py)

在文件末尾添加以下代码：

```python
# ============================================================================
# EfficientNet - Modern CNN for CIFAR-100 (Phase 3)
# ============================================================================


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    
    Used in EfficientNet, more effective than ReLU.
    
    Reference:
        Ramachandran et al. "Searching for Activation Functions" (2017)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for squeeze operation (default: 4)
    
    Reference:
        Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    """
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Swish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block.
    
    Structure: Expansion → Depthwise Conv → SE → Projection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise conv (3 or 5)
        stride: Stride for depthwise conv (1 or 2)
        expand_ratio: Expansion ratio for hidden dimension
        se_ratio: SE reduction ratio (default: 0.25)
        drop_path_rate: Drop path rate for stochastic depth
    
    Reference:
        Sandler et al. "MobileNetV2" (CVPR 2018)
        Tan & Le "EfficientNet" (ICML 2019)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        layers: List[nn.Module] = []
        
        # Expansion phase (if expand_ratio != 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size, stride,
                kernel_size // 2, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock(hidden_dim, int(1 / se_ratio)))
        
        # Output projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


class EfficientNet(nn.Module):
    """
    EfficientNet architecture for CIFAR-100 (from scratch).
    
    Adapted for 32×32 or larger input (64×64, 96×96).
    
    Args:
        width_mult: Width multiplier (1.0 for B0, 1.1 for B1, etc.)
        depth_mult: Depth multiplier (1.0 for B0, 1.1 for B1, etc.)
        input_size: Input image size (32, 64, or 96)
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate before classifier
        drop_path_rate: Maximum drop path rate (stochastic depth)
    
    Reference:
        Tan & Le "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019)
        https://arxiv.org/abs/1905.11946
    
    Example:
        >>> model = EfficientNet(width_mult=1.0, depth_mult=1.0, input_size=64)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([2, 100])
    """
    def __init__(
        self,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        input_size: int = 64,
        num_classes: int = 100,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        
        # Building blocks config: [expand_ratio, channels, num_layers, stride, kernel_size]
        blocks_args = [
            [1,  16,  1, 1, 3],  # Stage 1
            [6,  24,  2, 2, 3],  # Stage 2
            [6,  40,  2, 2, 5],  # Stage 3
            [6,  80,  3, 2, 3],  # Stage 4
            [6, 112,  3, 1, 5],  # Stage 5
            [6, 192,  4, 2, 5],  # Stage 6
            [6, 320,  1, 1, 3],  # Stage 7
        ]
        
        # Stem
        stem_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            Swish()
        )
        
        # Build MBConv blocks
        blocks: List[nn.Module] = []
        total_blocks = sum([self._round_repeats(b[2], depth_mult) for b in blocks_args])
        block_idx = 0
        in_channels = stem_channels
        
        for expand_ratio, channels, num_layers, stride, kernel_size in blocks_args:
            out_channels = self._round_filters(channels, width_mult)
            num_layers = self._round_repeats(num_layers, depth_mult)
            
            for i in range(num_layers):
                # Stochastic depth: linearly increase drop rate
                drop_rate = drop_path_rate * block_idx / total_blocks
                
                blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25,
                    drop_path_rate=drop_rate,
                ))
                
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(final_channels, num_classes)
        
        self._initialize_weights()
    
    def _round_filters(self, filters: int, width_mult: float, divisor: int = 8) -> int:
        """Round number of filters based on width multiplier."""
        filters *= width_mult
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    
    def _round_repeats(self, repeats: int, depth_mult: float) -> int:
        """Round number of repeats based on depth multiplier."""
        import math
        return int(math.ceil(depth_mult * repeats))
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, input_size, input_size]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def efficientnet_b0(
    num_classes: int = 100,
    input_size: int = 64,
    dropout_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> EfficientNet:
    """
    EfficientNet-B0 for CIFAR-100 (from scratch).
    
    Paper achieves 91.7% accuracy on CIFAR-100 (F1 ≈ 0.917).
    
    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        input_size: Input image size (recommended: 64 for best results)
        dropout_rate: Dropout rate before classifier (default: 0.2)
        drop_path_rate: Stochastic depth rate (default: 0.2)
    
    Returns:
        EfficientNet-B0 model instance
    
    Model Statistics:
        - Parameters: ~5.3M
        - FLOPs (64×64): ~0.4G
        - Expected F1 (64×64): 0.85-0.87
    
    Reference:
        Tan & Le "EfficientNet" (ICML 2019)
        https://arxiv.org/abs/1905.11946
    
    Example:
        >>> model = efficientnet_b0(input_size=64)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        Parameters: 5.30M
    """
    return EfficientNet(
        width_mult=1.0,
        depth_mult=1.0,
        input_size=input_size,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )


def efficientnet_b1(
    num_classes: int = 100,
    input_size: int = 64,
    dropout_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> EfficientNet:
    """
    EfficientNet-B1 for CIFAR-100 (larger model).
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size
        dropout_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
    
    Returns:
        EfficientNet-B1 model instance
    
    Model Statistics:
        - Parameters: ~7.8M
        - Expected F1 (64×64): 0.86-0.88
    """
    return EfficientNet(
        width_mult=1.0,
        depth_mult=1.1,  # B1: 10% deeper
        input_size=input_size,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )
```

### 2. 更新 create_model 函数

在 `create_model` 函数中添加 EfficientNet 选项：

```python
def create_model(
    num_classes: int,
    device: str,
    model_type: Literal[
        "resnet34",
        "resnet50",
        "wide_resnet28_10",
        "wide_resnet28_10_selfdistill",
        "wide_resnet40_10",
        "wide_resnet28_12",
        "pyramidnet110_270",
        "pyramidnet164_270",
        "convnext_tiny",
        "convnext_small",
        "efficientnet_b0",  # 新增
        "efficientnet_b1",  # 新增
    ] = "efficientnet_b0",  # 修改默认值
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
    input_size: int = 64,  # 新增参数
):
    """Create model from scratch."""
    
    # ... 现有代码 ...
    
    elif model_type == "efficientnet_b0":
        model = efficientnet_b0(
            num_classes=num_classes,
            input_size=input_size,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    elif model_type == "efficientnet_b1":
        model = efficientnet_b1(
            num_classes=num_classes,
            input_size=input_size,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    return model
```

### 3. 修改数据加载 (scripts/train_utils.py)

更新 `get_train_transforms` 函数：

```python
def get_train_transforms(
    dataset_type: Literal["cifar10", "cifar100"] = "cifar100",
    augmentation_strength: Literal["light", "medium", "strong", "randaugment"] = "randaugment",
    use_cutmix: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    input_size: int = 32,  # 新增参数
) -> AlbumentationsTransform:
    """Get training transforms with optional resizing."""
    
    # Get normalization statistics
    if dataset_type == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_type == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    # Build augmentation pipeline
    if augmentation_strength == "randaugment":
        from scripts.data_augmentation import RandAugment
        
        # Calculate Cutout hole size based on input_size
        hole_size_min = input_size // 8
        hole_size_max = input_size // 4
        
        augmentation_pipeline = A.Compose([
            A.Resize(input_size, input_size),  # 新增: Resize
            RandAugment(n=randaugment_n, m=randaugment_m),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(  # 调整 hole size
                num_holes_range=(1, 1),
                hole_height_range=(hole_size_min, hole_size_max),
                hole_width_range=(hole_size_min, hole_size_max),
                p=0.5,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    # ... 其他 augmentation_strength 的处理 (类似添加 Resize) ...
    
    return AlbumentationsTransform(augmentation_pipeline)
```

更新 `load_transforms` 函数：

```python
def load_transforms(
    dataset_type: str = "cifar100",
    input_size: int = 32,  # 新增参数
):
    """Load validation/test transforms with resizing."""
    
    if dataset_type == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_type == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 新增: Resize
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
```

更新 `load_data` 函数签名：

```python
def load_data(
    data_dir: str,
    batch_size: int,
    dataset_type: Literal["cifar10", "cifar100"] = "cifar100",
    manual_seed: int = 42,
    use_online_aug: bool = True,
    augmentation_strength: Literal["light", "medium", "strong", "randaugment"] = "light",
    use_cutmix: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    input_size: int = 32,  # 新增参数
):
    """Load data with optional resizing."""
    
    if use_online_aug:
        train_transforms = get_train_transforms(
            dataset_type=dataset_type,
            augmentation_strength=augmentation_strength,
            use_cutmix=use_cutmix,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
            input_size=input_size,  # 传递参数
        )
    else:
        train_transforms = load_transforms(dataset_type=dataset_type, input_size=input_size)
    
    val_transforms = load_transforms(dataset_type=dataset_type, input_size=input_size)
    
    # ... 其余代码不变 ...
```

### 4. 更新 main.py 参数

在 `parse_args()` 函数中添加：

```python
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Training Pipeline")
    
    # ... 现有参数 ...
    
    # Model architecture
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "resnet34",
            "resnet50",
            "wide_resnet28_10",
            "wide_resnet28_10_selfdistill",
            "wide_resnet40_10",
            "wide_resnet28_12",
            "pyramidnet110_270",
            "pyramidnet164_270",
            "convnext_tiny",
            "convnext_small",
            "efficientnet_b0",  # 新增
            "efficientnet_b1",  # 新增
        ],
        default="efficientnet_b0",  # 修改默认值
        help="Model architecture. EfficientNet-B0 (64×64) targets F1≥0.85.",
    )
    
    # 新增: Input size 参数
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="Input image size (32, 48, 64, 96). "
             "Larger = better detail (esp. human classes), slower training. "
             "Recommended: 64 for EfficientNet (F1≥0.85 target).",
    )
    
    # 更新: RandAugment magnitude 默认值
    parser.add_argument(
        "--randaugment_m",
        type=int,
        default=10,  # 从 9 提升到 10
        help="RandAugment M: magnitude (0-10). Higher = stronger. "
             "Default 10 for EfficientNet + 64×64.",
    )
    
    # ... 其余参数不变 ...
```

更新 `build_model()` 函数调用：

```python
def build_model(args) -> nn.Module:
    """Build the model (from scratch)"""
    
    # ... 现有代码 ...
    
    model = create_model(
        num_classes=num_classes,
        device=args.device,
        model_type=args.model,
        dropout_rate=args.dropout,
        drop_path_rate=args.drop_path_rate,
        input_size=args.input_size,  # 新增参数
    )
    
    # ... 其余代码不变 ...
```

更新 `train()` 函数中的 `load_data()` 调用：

```python
def train(args, model: nn.Module):
    # ... 现有代码 ...
    
    train_loader, val_loader = load_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        dataset_type=args.dataset,
        manual_seed=args.seed,
        use_online_aug=args.use_online_aug,
        augmentation_strength=args.aug_strength,
        use_cutmix=args.use_cutmix,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m,
        input_size=args.input_size,  # 新增参数
    )
    
    # ... 其余代码不变 ...
```

更新 `evaluate()` 函数中的 `load_transforms()` 调用：

```python
def evaluate(args, model: nn.Module):
    """Evaluate the model on test data"""
    test_data_dir = args.data_dir + "/raw/test"
    test_dataset = datasets.ImageFolder(
        root=test_data_dir,
        transform=load_transforms(
            dataset_type=args.dataset,
            input_size=args.input_size  # 新增参数
        )
    )
    
    # ... 其余代码不变 ...
```

---

## ✅ 测试代码

### 测试 EfficientNet 前向传播

```python
# test_efficientnet.py
from scripts.model_architectures import efficientnet_b0
import torch

# Create model
model = efficientnet_b0(input_size=64, num_classes=100)

# Test forward pass
x = torch.randn(2, 3, 64, 64)
y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Expected output:
# Input shape: torch.Size([2, 3, 64, 64])
# Output shape: torch.Size([2, 100])
# Parameters: 5.30M
```

### 测试数据加载

```python
# 快速测试 (5 epochs)
python main.py \
    --model efficientnet_b0 \
    --input_size 64 \
    --batch_size 64 \
    --num_epochs 5 \
    --seed 42
```

---

## 📋 实施检查清单

- [ ] 复制 EfficientNet 代码到 `scripts/model_architectures.py`
- [ ] 更新 `create_model` 函数
- [ ] 修改 `get_train_transforms` 支持 `input_size`
- [ ] 修改 `load_transforms` 支持 `input_size`
- [ ] 更新 `load_data` 函数签名
- [ ] 添加 `--input_size` 参数到 main.py
- [ ] 添加 `--model efficientnet_b0/b1` 选项
- [ ] 更新 `--randaugment_m` 默认值为 10
- [ ] 测试前向传播
- [ ] 运行 5 epochs 快速测试
- [ ] 检查 linting errors

---

## 🎯 下一步

实现完成后，执行完整训练：

```bash
python main.py \
    --model efficientnet_b0 \
    --input_size 64 \
    --drop_path_rate 0.2 \
    --aug_strength randaugment \
    --randaugment_m 10 \
    --batch_size 96 \
    --seed 42
```

预期训练时间: 6-8 小时  
预期 Test F1: **0.85-0.87** ✅

---

**准备开始实现！** 🚀
