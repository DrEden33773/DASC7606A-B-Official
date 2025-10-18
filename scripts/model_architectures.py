from typing import List, Literal, Optional, Protocol, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision.models removed - no pretrained models allowed per assignment guidelines


# ============================================================================
# Stochastic Depth (DropPath) - Used by Wide ResNet
# ============================================================================


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.

    Randomly drops entire residual branches during training to improve
    generalization and reduce overfitting. This is a key technique for
    training deep Wide ResNets.

    Args:
        drop_prob: Probability of dropping the path (0.0-1.0)
                   - 0.0: No dropout (identity)
                   - 0.1-0.2: Recommended for Wide ResNet

    References:
        Huang et al. "Deep Networks with Stochastic Depth" (ECCV 2016)
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor

        return output


# ============================================================================
# ResNet Block Protocol (for type checking)
# ============================================================================


class ResNetBlock(Protocol):
    """Protocol for ResNet block types (BasicBlock, Bottleneck, etc.)"""

    expansion: int

    def __call__(self, x): ...

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None: ...


# SimpleCNN removed - no longer needed for CIFAR-100 optimization
# Use ResNet or Wide ResNet variants instead for better performance


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (1 or 2)
        downsample: Optional downsampling layer for skip connection
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet (used in ResNet-50/101/152)

    This block uses a 1x1 -> 3x3 -> 1x1 convolution structure:
    1. 1x1 conv reduces channels (compression)
    2. 3x3 conv processes features in lower dimensional space
    3. 1x1 conv expands channels back (expansion)

    This design is more parameter-efficient than BasicBlock for deeper networks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (before expansion)
        stride: Stride for the 3x3 convolution (1 or 2)
        downsample: Optional downsampling layer for skip connection

    Note:
        The actual output channels will be out_channels * expansion (expansion = 4)
    """

    expansion = 4  # Output channels are 4x the base channels

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(Bottleneck, self).__init__()

        # 1x1 convolution for channel reduction (compression)
        # Example: 256 -> 64 channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution (processes features in compressed space)
        # Example: 64 -> 64 channels
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution for channel expansion
        # Example: 64 -> 256 channels (64 * expansion = 64 * 4)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 compression
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 3x3 convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # 1x1 expansion
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection (with optional downsampling)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet architecture optimized for CIFAR-10/100 (32x32 images)

    Modified from the original ResNet to work with smaller input sizes:
    - Uses 3x3 conv instead of 7x7 for the initial layer
    - Removes the initial max pooling layer
    - Adapted for 32x32 input images

    Args:
        block: Type of residual block (BasicBlock or Bottleneck)
        layers: List of number of blocks in each layer
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (default: 0.3)
    """

    def __init__(
        self,
        block: Type[ResNetBlock],
        layers: list,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ):
        super(ResNetCIFAR, self).__init__()

        self.in_channels = 64

        # Initial convolution layer (adapted for CIFAR 32x32 input)
        # Using 3x3 conv instead of 7x7 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No max pooling here (unlike standard ResNet) to preserve 32x32 resolution

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[ResNetBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Create a residual layer consisting of multiple blocks

        Args:
            block: Type of residual block
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for the first block

        Returns:
            Sequential layer containing all blocks
        """
        downsample = None

        # Create downsampling layer if needed (when stride != 1 or channels change)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block (may have downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # No max pooling for CIFAR

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


# resnet18_cifar removed - not used in experiments, no performance benefit over resnet34


def resnet34_cifar(num_classes: int = 10, dropout_rate: float = 0.3) -> ResNetCIFAR:
    """
    Construct ResNet-34 model for CIFAR datasets

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization

    Returns:
        ResNet-34 model optimized for CIFAR
    """
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes, dropout_rate)


def resnet50_cifar(num_classes: int = 10, dropout_rate: float = 0.3) -> ResNetCIFAR:
    """
    Construct ResNet-50 model for CIFAR datasets

    This model uses Bottleneck blocks instead of BasicBlock:
    - 50 layers total (vs 34 for ResNet-34)
    - Uses 1x1->3x3->1x1 convolution structure (more parameter-efficient)
    - 4x expansion in each block (vs 1x for BasicBlock)
    - More representational capacity for complex datasets like CIFAR-100

    Architecture: [3, 4, 6, 3] blocks in 4 stages
    - Stage 1: 64 channels  -> 256 channels (3 blocks)
    - Stage 2: 128 channels -> 512 channels (4 blocks)
    - Stage 3: 256 channels -> 1024 channels (6 blocks)
    - Stage 4: 512 channels -> 2048 channels (3 blocks)

    Total parameters: ~23.5M (vs ~21M for ResNet-34)

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate for regularization (default: 0.3)

    Returns:
        ResNet-50 model optimized for CIFAR

    Note:
        For CIFAR-100, recommended dropout_rate is 0.4-0.5 due to increased model capacity.
        Expected performance: +0.02-0.03 F1 improvement over ResNet-34 (from scratch).
    """
    return ResNetCIFAR(Bottleneck, [3, 4, 6, 3], num_classes, dropout_rate)


# create_pretrained_resnet removed - pretraining is not allowed for this assignment
# All models must be trained from scratch per assignment guidelines


# ============================================================================
# Wide ResNet - Optimized for CIFAR-100 from scratch training
# ============================================================================


class WideBasicBlock(nn.Module):
    """Wide ResNet basic residual block with pre-activation structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_path = DropPath(drop_path_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.drop_path(out)
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    Wide Residual Network for CIFAR datasets.

    Achieved F1=0.8131 on CIFAR-100 with drop_path_rate=0.1 + RandAugment.
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        assert (depth - 4) % 6 == 0, f"Depth must satisfy (depth-4)%6==0, got {depth}"

        n_blocks = (depth - 4) // 6
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Calculate stochastic depth rates (linear increase)
        total_blocks = n_blocks * 3
        drop_rates = [
            i * drop_path_rate / (total_blocks - 1) if total_blocks > 1 else 0.0
            for i in range(total_blocks)
        ]

        self.in_channels = n_channels[0]
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        block_idx = 0
        self.layer1 = self._make_layer(
            n_channels[1],
            n_blocks,
            dropout_rate,
            stride=1,
            drop_rates=drop_rates[block_idx : block_idx + n_blocks],
        )
        block_idx += n_blocks
        self.layer2 = self._make_layer(
            n_channels[2],
            n_blocks,
            dropout_rate,
            stride=2,
            drop_rates=drop_rates[block_idx : block_idx + n_blocks],
        )
        block_idx += n_blocks
        self.layer3 = self._make_layer(
            n_channels[3],
            n_blocks,
            dropout_rate,
            stride=2,
            drop_rates=drop_rates[block_idx : block_idx + n_blocks],
        )

        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.9)
        self.fc = nn.Linear(n_channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        drop_rates: List[float],
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []

        for i, stride in enumerate(strides):
            layers.append(
                WideBasicBlock(
                    self.in_channels,
                    out_channels,
                    stride,
                    dropout_rate,
                    drop_path_rate=drop_rates[i],
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def wide_resnet28_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """Wide ResNet-28-10 for CIFAR (36.5M params). Best: F1=0.8131 with drop_path=0.1."""
    return WideResNet(28, 10, num_classes, dropout_rate, drop_path_rate)


def wide_resnet40_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """Wide ResNet-40-10 for CIFAR (55.8M params)."""
    return WideResNet(40, 10, num_classes, dropout_rate, drop_path_rate)


def wide_resnet28_12(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """Wide ResNet-28-12 for CIFAR (52.8M params)."""
    return WideResNet(28, 12, num_classes, dropout_rate, drop_path_rate)


def create_model(
    num_classes: int,
    device: str,
    model_type: Literal[
        "resnet34",
        "resnet50",
        "wide_resnet28_10",
        "wide_resnet40_10",
        "wide_resnet28_12",
    ] = "wide_resnet28_10",
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
):
    """
    Create and initialize the model from scratch.

    All models are trained from scratch (no pretraining) per assignment guidelines.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        device: Device to place the model on ('cuda' or 'cpu')
        model_type: Type of model architecture to use. Options:
            ResNet variants (baseline):
            - "resnet34": ResNet-34 (21M params, F1=0.77)
            - "resnet50": ResNet-50 with Bottleneck (23.5M params, F1=0.77)

            Wide ResNet variants (recommended, proven effective):
            - "wide_resnet28_10": WRN-28-10 (36.5M params, F1=0.8131) ← Best
            - "wide_resnet40_10": WRN-40-10 (55.8M params)
            - "wide_resnet28_12": WRN-28-12 (52.8M params)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        drop_path_rate: Stochastic depth rate (default: 0.0, best: 0.1 for WRN-28-10)

    Returns:
        Model instance moved to the specified device

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> # Best configuration (F1=0.8131)
        >>> model = create_model(100, 'cuda', 'wide_resnet28_10',
        ...                       dropout_rate=0.3, drop_path_rate=0.1)

    Note:
        Wide ResNet-28-10 with drop_path=0.1 + RandAugment achieved F1=0.8131,
        significantly outperforming ResNet variants (+0.04).
    """
    # Create model from scratch
    if model_type == "resnet34":
        model = resnet34_cifar(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "resnet50":
        model = resnet50_cifar(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "wide_resnet28_10":
        model = wide_resnet28_10(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    elif model_type == "wide_resnet40_10":
        model = wide_resnet40_10(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    elif model_type == "wide_resnet28_12":
        model = wide_resnet28_12(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'resnet34', 'resnet50', "
            f"'wide_resnet28_10', 'wide_resnet40_10', 'wide_resnet28_12'"
        )

    model = model.to(device)

    return model
