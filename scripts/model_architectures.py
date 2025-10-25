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
# PyramidNet - Deep Pyramidal Residual Networks (Phase 2.7)
# ============================================================================


class PyramidBasicBlock(nn.Module):
    """
    PyramidNet basic block with pre-activation and zero-padded shortcuts.

    Reference:
        Han et al. "Deep Pyramidal Residual Networks" (CVPR 2017)
        https://arxiv.org/abs/1610.02915
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
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
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_path = DropPath(drop_path_rate)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut path (before pre-activation!)
        shortcut = x

        # Downsampling if needed
        if self.stride != 1:
            shortcut = F.avg_pool2d(shortcut, kernel_size=2, stride=2)

        # Zero-pad channels if dimension increases (PyramidNet key technique)
        if self.in_channels != self.out_channels:
            pad_channels = self.out_channels - self.in_channels
            # Pad on channel dimension: [N, C, H, W] -> [N, C+pad, H, W]
            shortcut = F.pad(
                shortcut,
                (0, 0, 0, 0, 0, pad_channels),
                mode="constant",
                value=0,
            )

        # Main path with pre-activation
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        # Drop path
        out = self.drop_path(out)

        # Add shortcut
        out = out + shortcut

        return out


class PyramidNet(nn.Module):
    """
    PyramidNet for CIFAR datasets.

    Gradually increases channel dimensions across all blocks instead of
    sudden jumps at downsampling points, improving feature diversity.

    Args:
        depth: Network depth (e.g., 110, 164, 272)
        alpha: Widening factor controlling final channel dimension
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        depth: int,
        alpha: int,
        num_classes: int = 100,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # Calculate blocks per group
        # PyramidNet depth = 2 + 6n (for bottleneck: 2 + 9n)
        # For BasicBlock: n = (depth - 2) / 6
        assert (depth - 2) % 6 == 0, f"Depth must satisfy (depth-2)%6==0, got {depth}"
        n = (depth - 2) // 6

        # Initial channels
        start_channels = 16

        # Calculate channel increments for each block
        # Total blocks: 3n (3 groups × n blocks)
        total_blocks = 3 * n
        add_channels = alpha / total_blocks  # Increment per block

        # Build channel list for each block
        in_channels_list = []
        out_channels_list = []

        current_channels = start_channels
        for i in range(total_blocks):
            in_channels_list.append(int(round(current_channels)))
            current_channels += add_channels
            out_channels_list.append(int(round(current_channels)))

        # Calculate stochastic depth rates (linear increase)
        dp_rates = [
            i * drop_path_rate / (total_blocks - 1) if total_blocks > 1 else 0.0
            for i in range(total_blocks)
        ]

        # Initial convolution (no BN here, PyramidBasicBlock has pre-activation)
        self.conv1 = nn.Conv2d(
            3, start_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Build three groups
        block_idx = 0
        self.layer1 = self._make_layer(
            n,
            in_channels_list[block_idx : block_idx + n],
            out_channels_list[block_idx : block_idx + n],
            dp_rates[block_idx : block_idx + n],
            stride=1,
        )
        block_idx += n

        self.layer2 = self._make_layer(
            n,
            in_channels_list[block_idx : block_idx + n],
            out_channels_list[block_idx : block_idx + n],
            dp_rates[block_idx : block_idx + n],
            stride=2,
        )
        block_idx += n

        self.layer3 = self._make_layer(
            n,
            in_channels_list[block_idx : block_idx + n],
            out_channels_list[block_idx : block_idx + n],
            dp_rates[block_idx : block_idx + n],
            stride=2,
        )

        # Final BN and FC
        final_channels = out_channels_list[-1]
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.fc = nn.Linear(final_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        num_blocks: int,
        in_channels_list: List[int],
        out_channels_list: List[int],
        dp_rates: List[float],
        stride: int,
    ) -> nn.Sequential:
        """Create a layer with gradually increasing channels."""
        layers: List[nn.Module] = []

        for i in range(num_blocks):
            # First block of group has stride, others have stride=1
            block_stride = stride if i == 0 else 1

            layers.append(
                PyramidBasicBlock(
                    in_channels=in_channels_list[i],
                    out_channels=out_channels_list[i],
                    stride=block_stride,
                    drop_path_rate=dp_rates[i],
                )
            )

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
        # Initial conv (no activation, pre-activation is in blocks)
        x = self.conv1(x)

        # Three pyramid layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Final activation and pooling
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def pyramidnet110_270(
    num_classes: int = 100,
    drop_path_rate: float = 0.1,
) -> PyramidNet:
    """
    PyramidNet-110 with alpha=270 for CIFAR.

    Paper result: CIFAR-100 ~83% accuracy (~F1 0.83)
    Target with SD+RA: F1 ≥ 0.85

    Architecture:
    - Depth: 110 layers
    - Alpha: 270 (widening factor)
    - Parameters: ~26M
    - Final channels: 16 + 270 = 286
    """
    return PyramidNet(
        depth=110, alpha=270, num_classes=num_classes, drop_path_rate=drop_path_rate
    )


def pyramidnet164_270(
    num_classes: int = 100,
    drop_path_rate: float = 0.15,
) -> PyramidNet:
    """
    PyramidNet-164 with alpha=270 for CIFAR (deeper variant).

    Phase 3 option if PyramidNet-110 is successful.
    """
    return PyramidNet(
        depth=164, alpha=270, num_classes=num_classes, drop_path_rate=drop_path_rate
    )


# ============================================================================
# Wide ResNet - Optimized for CIFAR-100 from scratch training (Phase 1)
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


class WideResNetSelfDistill(WideResNet):
    """
    Wide ResNet with Self-Distillation (Be Your Own Teacher).

    Adds intermediate classifiers after each layer group for self-distillation.
    During training, deep layers teach shallow layers. During inference, only
    the final classifier is used.

    Reference:
        Zhang et al. "Be Your Own Teacher: Improve the Performance of
        Convolutional Neural Networks via Self Distillation" (2019)
        https://arxiv.org/abs/1905.08094
    """

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
    ) -> None:
        # Initialize base Wide ResNet
        super().__init__(depth, widen_factor, num_classes, dropout_rate, drop_path_rate)

        # Calculate channel dimensions
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Create intermediate classifiers (student classifiers)
        self.classifier1 = self._make_auxiliary_classifier(n_channels[1], num_classes)
        self.classifier2 = self._make_auxiliary_classifier(n_channels[2], num_classes)
        self.classifier3 = self._make_auxiliary_classifier(n_channels[3], num_classes)

    def _make_auxiliary_classifier(
        self, in_channels: int, num_classes: int
    ) -> nn.Module:
        """
        Create auxiliary classifier for intermediate supervision.

        Uses bottleneck structure to reduce interference with main network.
        """
        return nn.Sequential(
            # Bottleneck: reduce channels by half
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Classifier
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass with optional intermediate outputs.

        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            return_all: If True, return all classifier outputs (for training)
                        If False, return only final output (for inference)

        Returns:
            If return_all=False: Final logits [batch_size, num_classes]
            If return_all=True: [logits1, logits2, logits3, logits4]
        """
        # Initial conv
        out = self.conv1(x)

        # Section 1: layer1
        out = self.layer1(out)
        logits1 = self.classifier1(out)

        # Section 2: layer2
        out = self.layer2(out)
        logits2 = self.classifier2(out)

        # Section 3: layer3
        out = self.layer3(out)
        logits3 = self.classifier3(out)

        # Final classifier (teacher)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        logits4 = self.fc(out)

        if return_all:
            return [logits1, logits2, logits3, logits4]
        else:
            return logits4


def wide_resnet28_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """Wide ResNet-28-10 for CIFAR (36.5M params). Best: F1=0.8131 with drop_path=0.1."""
    return WideResNet(28, 10, num_classes, dropout_rate, drop_path_rate)


def wide_resnet28_10_selfdistill(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNetSelfDistill:
    """
    Wide ResNet-28-10 with Self-Distillation (BYOT).

    Target: F1 ≥ 0.85 (based on +3-4% improvement from paper)
    """
    return WideResNetSelfDistill(28, 10, num_classes, dropout_rate, drop_path_rate)


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


# ============================================================================
# Ensemble Model - Combines multiple models for ensemble prediction
# ============================================================================


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines multiple trained models via soft voting.

    Acts as a single model but internally runs multiple models and averages predictions.
    Compatible with standard evaluate() function.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        """
        Initialize ensemble model.

        Args:
            models: List of trained models (should all have same architecture)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (soft voting).

        Args:
            x: Input tensor [batch_size, C, H, W]

        Returns:
            Averaged logits [batch_size, num_classes]
        """
        # Get predictions from all models
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)

        # Average logits (soft voting)
        ensemble_output = torch.stack(outputs).mean(dim=0)

        return ensemble_output

    def train(self, mode: bool = True):
        """Set ensemble to train/eval mode."""
        super().train(mode)
        for model in self.models:
            model.train(mode)
        return self

    def eval(self):
        """Set ensemble to eval mode."""
        return self.train(False)


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
            Phase 1 (proven):
            - "wide_resnet28_10": WRN-28-10 (36.5M params, F1=0.8131) ← Phase 1 best

            Phase 2.7 (PyramidNet, target F1≥0.85):
            - "pyramidnet110_270": PyramidNet-110 (26M params, paper: 83% acc) ← Recommended
            - "pyramidnet164_270": PyramidNet-164 (26M params) ← Deeper variant

            Phase 2.5 (self-distillation):
            - "wide_resnet28_10_selfdistill": WRN-28-10 + BYOT (39M params, F1=0.7968)

            Others:
            - "wide_resnet40_10": WRN-40-10 (55.8M params)
            - "wide_resnet28_12": WRN-28-12 (52.8M params, F1=0.80)
            - "resnet34": ResNet-34 (21M params, F1=0.77)
            - "resnet50": ResNet-50 (23.5M params, F1=0.77)
        dropout_rate: Dropout rate (only for ResNet/Wide ResNet)
        drop_path_rate: Stochastic depth rate (best: 0.1 for WRN)

    Returns:
        Model instance moved to the specified device

    Example:
        >>> # Phase 1 best (F1=0.8131)
        >>> model = create_model(100, 'cuda', 'wide_resnet28_10', drop_path_rate=0.1)
        >>>
        >>> # Phase 2.7 target
        >>> model = create_model(100, 'cuda', 'pyramidnet110_270', drop_path_rate=0.15)
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
    elif model_type == "wide_resnet28_10_selfdistill":
        model = wide_resnet28_10_selfdistill(
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
    elif model_type == "pyramidnet110_270":
        model = pyramidnet110_270(
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
    elif model_type == "pyramidnet164_270":
        model = pyramidnet164_270(
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'resnet34', 'resnet50', "
            f"'wide_resnet28_10', 'wide_resnet28_10_selfdistill', "
            f"'wide_resnet40_10', 'wide_resnet28_12', "
            f"'pyramidnet110_270', 'pyramidnet164_270'"
        )

    model = model.to(device)

    return model
