from typing import List, Literal, Optional, Protocol, Type, Union

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

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for deeper ResNets (ResNet-50/101/152)

    Uses 1x1 -> 3x3 -> 1x1 convolutions to reduce parameters

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the 3x3 convolution (1 or 2)
        downsample: Optional downsampling layer for skip connection
    """

    expansion = 4  # Output channels = out_channels * expansion

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(Bottleneck, self).__init__()

        # 1x1 convolution (reduce dimensions)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution (main computation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution (restore dimensions)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Add skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture adapted for CIFAR-100

    This implementation is optimized for 32x32 images:
    - Uses 3x3 conv with stride=1 as the first layer (instead of 7x7 with stride=2)
    - Removed the first MaxPool layer
    - Standard residual blocks with skip connections

    Args:
        block: Type of residual block (BasicBlock or Bottleneck)
        layers: List of number of blocks in each stage
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate before final classifier
    """

    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        layers: List[int],
        num_classes: int = 100,
        dropout_rate: float = 0.3,
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolution (adapted for CIFAR-100's 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int,
    ):
        downsample = None

        # Create downsampling layer if needed (stride != 1 or channel mismatch)
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

        layers: List[nn.Module] = []

        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet34_cifar(num_classes: int = 100, dropout_rate: float = 0.3) -> ResNet:
    """
    ResNet-34 adapted for CIFAR-100

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate before final classifier

    Returns:
        ResNet-34 model instance

    Model size: ~21M parameters
    Expected F1-score: ~0.77 (Phase 1)
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dropout_rate)


def resnet50_cifar(num_classes: int = 100, dropout_rate: float = 0.3) -> ResNet:
    """
    ResNet-50 adapted for CIFAR-100

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate before final classifier

    Returns:
        ResNet-50 model instance

    Model size: ~23.5M parameters
    Expected F1-score: ~0.77 (Phase 1)
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout_rate)


# ============================================================================
# Wide ResNet - High capacity variant of ResNet
# ============================================================================


class WideBasicBlock(nn.Module):
    """
    Wide Basic Block for Wide ResNet with Stochastic Depth support.

    Compared to standard BasicBlock:
    - Supports width multiplier (k) to increase channel capacity
    - Adds dropout for regularization
    - Supports stochastic depth (DropPath) for better generalization
    - Modified activation order (BN-ReLU-Conv)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution (1 or 2)
        dropout_rate: Dropout probability between conv layers
        drop_path_rate: Stochastic depth drop probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
    ):
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
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Stochastic depth (DropPath)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(x))
        out = self.conv1(out)

        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        # Apply stochastic depth
        out = self.drop_path(out)

        # Skip connection
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """
    Wide ResNet architecture for CIFAR-100.

    Wide ResNet achieves better accuracy than standard ResNet by:
    1. Increasing width (channels) instead of depth
    2. Using dropout for regularization
    3. Supporting stochastic depth for deeper networks

    Args:
        depth: Total network depth (e.g., 28 for WRN-28-10)
        widen_factor: Width multiplier (e.g., 10 for WRN-28-10)
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate between conv layers
        drop_path_rate: Stochastic depth rate (recommended: 0.1-0.2)

    Architecture:
        - Initial conv: 3 -> 16 channels
        - Group 1: 16 -> 16*k channels, spatial size 32x32
        - Group 2: 16*k -> 32*k channels, spatial size 16x16
        - Group 3: 32*k -> 64*k channels, spatial size 8x8
        - Global pooling + classifier

    Reference:
        Zagoruyko & Komodakis "Wide Residual Networks" (BMVC 2016)
        https://arxiv.org/abs/1605.07146
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Calculate number of blocks per group
        assert (depth - 4) % 6 == 0, "depth should be 6n+4 (e.g., 28, 40)"
        n = (depth - 4) // 6  # Number of blocks per group

        # Channel progression
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Initial convolution
        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # Calculate stochastic depth rates (linearly increasing)
        total_blocks = 3 * n
        drop_rates = [drop_path_rate * i / total_blocks for i in range(total_blocks)]

        # Three groups of wide residual blocks
        self.group1 = self._make_group(
            WideBasicBlock,
            channels[0],
            channels[1],
            n,
            stride=1,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[0:n],
        )
        self.group2 = self._make_group(
            WideBasicBlock,
            channels[1],
            channels[2],
            n,
            stride=2,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[n : 2 * n],
        )
        self.group3 = self._make_group(
            WideBasicBlock,
            channels[2],
            channels[3],
            n,
            stride=2,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[2 * n :],
        )

        # Final layers
        self.bn = nn.BatchNorm2d(channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_group(
        self,
        block: Type[WideBasicBlock],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
        drop_rates: List[float],
    ):
        layers = []

        # First block (may downsample)
        layers.append(
            block(in_channels, out_channels, stride, dropout_rate, drop_rates[0])
        )

        # Remaining blocks
        for i in range(1, num_blocks):
            layers.append(
                block(out_channels, out_channels, 1, dropout_rate, drop_rates[i])
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def wide_resnet28_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """
    Wide ResNet-28-10 for CIFAR-100.

    Best performing model from Phase 1.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate (recommended: 0.3)
        drop_path_rate: Stochastic depth rate (recommended: 0.1)

    Returns:
        WRN-28-10 model instance

    Model Statistics:
        - Parameters: ~36.5M
        - Expected F1: 0.8131 (Phase 1 best)
        - Training time: ~8-10 hours (600 epochs)

    Reference:
        Phase 1 best result: F1=0.8131 with dropout=0.3, drop_path=0.1
    """
    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )


def wide_resnet40_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """
    Wide ResNet-40-10 for CIFAR-100.

    Deeper variant with potentially better capacity.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate (recommended: 0.3)
        drop_path_rate: Stochastic depth rate (recommended: 0.1-0.2)

    Returns:
        WRN-40-10 model instance

    Model Statistics:
        - Parameters: ~55.8M
        - Deeper than WRN-28-10 but similar width
    """
    return WideResNet(
        depth=40,
        widen_factor=10,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )


def wide_resnet28_12(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """
    Wide ResNet-28-12 for CIFAR-100.

    Wider variant with more capacity than WRN-28-10.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate (recommended: 0.3)
        drop_path_rate: Stochastic depth rate (recommended: 0.1)

    Returns:
        WRN-28-12 model instance

    Model Statistics:
        - Parameters: ~52.8M
        - Expected F1: 0.82 (Phase 1)
    """
    return WideResNet(
        depth=28,
        widen_factor=12,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )


# ============================================================================
# Self-Distillation: Wide ResNet with auxiliary classifiers
# ============================================================================


class WideResNetSelfDistill(nn.Module):
    """
    Wide ResNet with self-distillation (BYOT) support.

    Adds auxiliary classifiers at intermediate layers for self-distillation training.
    This variant did not improve performance (F1=0.7968 vs 0.8131 for standard WRN).

    Args:
        depth: Network depth
        widen_factor: Width multiplier
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        drop_path_rate: Stochastic depth rate

    Reference:
        Zhang et al. "Be Your Own Teacher" (ICLR 2020)
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 100,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Calculate number of blocks per group
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6

        # Channel progression
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Initial convolution
        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # Calculate stochastic depth rates
        total_blocks = 3 * n
        drop_rates = [drop_path_rate * i / total_blocks for i in range(total_blocks)]

        # Three groups of wide residual blocks
        self.group1 = self._make_group(
            WideBasicBlock,
            channels[0],
            channels[1],
            n,
            stride=1,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[0:n],
        )
        self.group2 = self._make_group(
            WideBasicBlock,
            channels[1],
            channels[2],
            n,
            stride=2,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[n : 2 * n],
        )
        self.group3 = self._make_group(
            WideBasicBlock,
            channels[2],
            channels[3],
            n,
            stride=2,
            dropout_rate=dropout_rate,
            drop_rates=drop_rates[2 * n :],
        )

        # Final layers
        self.bn = nn.BatchNorm2d(channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

        # Auxiliary classifiers for self-distillation
        self.aux1 = self._make_auxiliary_classifier(channels[1], num_classes)
        self.aux2 = self._make_auxiliary_classifier(channels[2], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_group(
        self,
        block: Type[WideBasicBlock],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
        drop_rates: List[float],
    ):
        layers = []
        layers.append(
            block(in_channels, out_channels, stride, dropout_rate, drop_rates[0])
        )
        for i in range(1, num_blocks):
            layers.append(
                block(out_channels, out_channels, 1, dropout_rate, drop_rates[i])
            )
        return nn.Sequential(*layers)

    def _make_auxiliary_classifier(self, in_channels: int, num_classes: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)

        # Group 1 with auxiliary output
        x = self.group1(x)
        aux1_out = self.aux1(x)

        # Group 2 with auxiliary output
        x = self.group2(x)
        aux2_out = self.aux2(x)

        # Group 3 and final output
        x = self.group3(x)
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        main_out = self.fc(x)

        # Return main and auxiliary outputs
        if self.training:
            return main_out, aux1_out, aux2_out
        else:
            return main_out


def wide_resnet28_10_selfdistill(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNetSelfDistill:
    """
    Wide ResNet-28-10 with self-distillation support.

    Note: This variant did not improve performance in Phase 2.5.
    Standard WRN-28-10 achieved F1=0.8131 vs F1=0.7968 for this variant.

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        drop_path_rate: Stochastic depth rate

    Returns:
        WRN-28-10 with auxiliary classifiers
    """
    return WideResNetSelfDistill(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )


# ============================================================================
# PyramidNet - Gradually increasing channel dimensions
# ============================================================================


class PyramidBasicBlock(nn.Module):
    """
    Basic block for PyramidNet with gradually increasing channels.

    Unlike ResNet which increases channels abruptly, PyramidNet gradually
    increases the number of channels throughout the network.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
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

        self.bn3 = nn.BatchNorm2d(out_channels)

        # Stochastic depth
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

        # Skip connection with channel adjustment if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride)
                if stride != 1
                else nn.Identity()
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)

        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.bn3(out)

        # Apply stochastic depth
        out = self.drop_path(out)

        # Handle skip connection with channel padding if needed
        shortcut = self.shortcut(x)
        if out.size(1) != shortcut.size(1):
            # Pad channels with zeros to match
            padding = torch.zeros(
                shortcut.size(0),
                out.size(1) - shortcut.size(1),
                shortcut.size(2),
                shortcut.size(3),
                device=shortcut.device,
                dtype=shortcut.dtype,
            )
            shortcut = torch.cat([shortcut, padding], dim=1)

        out += shortcut
        return out


class PyramidNet(nn.Module):
    """
    PyramidNet architecture for CIFAR-100.

    PyramidNet gradually increases the number of channels throughout the network,
    rather than increasing them abruptly like ResNet.

    Args:
        depth: Total network depth
        alpha: Widening factor (determines final channel count)
        num_classes: Number of output classes
        drop_path_rate: Maximum stochastic depth rate

    Architecture:
        - Initial channels: 16
        - Gradually increases to: 16 + alpha
        - Three groups with different spatial resolutions

    Reference:
        Han et al. "Deep Pyramidal Residual Networks" (CVPR 2017)
        https://arxiv.org/abs/1610.02915
    """

    def __init__(
        self,
        depth: int = 110,
        alpha: int = 270,
        num_classes: int = 100,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Calculate number of blocks per group
        assert (depth - 2) % 6 == 0, "depth should be 6n+2 (e.g., 110, 164)"
        n = (depth - 2) // 6

        # Initial channels
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Calculate channel increments
        add_per_block = alpha / (3 * n)

        # Calculate stochastic depth rates
        total_blocks = 3 * n
        drop_rates = [drop_path_rate * i / total_blocks for i in range(total_blocks)]

        # Three groups
        self.layer1 = self._make_layer(
            PyramidBasicBlock,
            n,
            add_per_block,
            stride=1,
            drop_rates=drop_rates[0:n],
        )
        self.layer2 = self._make_layer(
            PyramidBasicBlock,
            n,
            add_per_block,
            stride=2,
            drop_rates=drop_rates[n : 2 * n],
        )
        self.layer3 = self._make_layer(
            PyramidBasicBlock,
            n,
            add_per_block,
            stride=2,
            drop_rates=drop_rates[2 * n :],
        )

        # Final layers
        self.bn_final = nn.BatchNorm2d(int(round(16 + alpha)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(round(16 + alpha)), num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[PyramidBasicBlock],
        num_blocks: int,
        add_per_block: float,
        stride: int,
        drop_rates: List[float],
    ):
        layers = []

        for i in range(num_blocks):
            # Calculate output channels for this block
            out_channels = int(round(self.in_channels + add_per_block))

            # Add block
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    drop_rates[i],
                )
            )

            # Update in_channels for next block
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def pyramidnet110_270(
    num_classes: int = 100,
    drop_path_rate: float = 0.0,
) -> PyramidNet:
    """
    PyramidNet-110 (alpha=270) for CIFAR-100.

    Args:
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate (recommended: 0.15)

    Returns:
        PyramidNet-110 model instance

    Model Statistics:
        - Parameters: ~26M
        - Paper reported accuracy: 83% on CIFAR-100
    """
    return PyramidNet(
        depth=110,
        alpha=270,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


def pyramidnet164_270(
    num_classes: int = 100,
    drop_path_rate: float = 0.0,
) -> PyramidNet:
    """
    PyramidNet-164 (alpha=270) for CIFAR-100.

    Deeper variant with potentially better capacity.

    Args:
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate (recommended: 0.15)

    Returns:
        PyramidNet-164 model instance

    Model Statistics:
        - Parameters: ~26M
        - Deeper than PyramidNet-110
    """
    return PyramidNet(
        depth=164,
        alpha=270,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


# ============================================================================
# Ensemble Models
# ============================================================================


class EnsembleModel(nn.Module):
    """
    Ensemble multiple models for improved prediction.

    Combines predictions from multiple models by averaging their outputs.
    Used in Phase 1 for final submission.

    Args:
        models: List of model instances to ensemble

    Example:
        >>> model1 = wide_resnet28_10()
        >>> model2 = wide_resnet28_12()
        >>> model3 = pyramidnet110_270()
        >>> ensemble = EnsembleModel([model1, model2, model3])
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        """Average predictions from all models."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Average the outputs
        return torch.stack(outputs).mean(dim=0)

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
        "convmixer_768_32",
        "convmixer_1536_20",
        "convmixer_1024_20",
    ] = "wide_resnet28_10",
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
    input_size: int = 32,
):
    """
    Create and initialize the model from scratch.

    All models are trained from scratch (no pretraining) per assignment guidelines.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        device: Device to place the model on ('cuda' or 'cpu')
        model_type: Type of model architecture to use. Options:
            Phase 3 (ConvMixer - NEW, target F1≥0.85):
            - "convmixer_768_32": ConvMixer-768/32 (21M params) ← Recommended
            - "convmixer_1536_20": ConvMixer-1536/20 (52M params, highest capacity)
            - "convmixer_1024_20": ConvMixer-1024/20 (24M params, balanced)

            Phase 1 (proven):
            - "wide_resnet28_10": WRN-28-10 (36.5M params, F1=0.8131) ← Phase 1 best
            - "wide_resnet28_12": WRN-28-12 (52.8M params, F1=0.82)

            Phase 2.7 (PyramidNet):
            - "pyramidnet110_270": PyramidNet-110 (26M params, paper: 83% acc)
            - "pyramidnet164_270": PyramidNet-164 (26M params) ← Deeper variant

            Phase 2.5 (self-distillation):
            - "wide_resnet28_10_selfdistill": WRN-28-10 + BYOT (39M params, F1=0.7968)

            Others:
            - "wide_resnet40_10": WRN-40-10 (55.8M params)
            - "resnet34": ResNet-34 (21M params, F1=0.77)
            - "resnet50": ResNet-50 (23.5M params, F1=0.77)
        dropout_rate: Dropout rate (ignored by ConvMixer and PyramidNet)
        drop_path_rate: Stochastic depth rate (ignored by ConvMixer)
        input_size: Input image size (32 for all models)

    Returns:
        Model instance moved to the specified device

    Example:
        >>> # Phase 3 ConvMixer (target F1≥0.85)
        >>> model = create_model(100, 'cuda', 'convmixer_768_32')
        >>> # Phase 1 best (F1=0.8131)
        >>> model = create_model(100, 'cuda', 'wide_resnet28_10', drop_path_rate=0.1)
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
    elif model_type == "convmixer_768_32":
        model = convmixer_768_32(num_classes=num_classes)
    elif model_type == "convmixer_1536_20":
        model = convmixer_1536_20(num_classes=num_classes)
    elif model_type == "convmixer_1024_20":
        model = convmixer_1024_20(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'resnet34', 'resnet50', "
            f"'wide_resnet28_10', 'wide_resnet28_10_selfdistill', "
            f"'wide_resnet40_10', 'wide_resnet28_12', "
            f"'pyramidnet110_270', 'pyramidnet164_270', "
            f"'convmixer_768_32', 'convmixer_1536_20', 'convmixer_1024_20'"
        )

    model = model.to(device)

    return model


# ============================================================================
# ConvMixer - Simple Yet Effective Patch-based Architecture (Phase 3)
# ============================================================================


class Residual(nn.Module):
    """
    Residual wrapper that applies fn(x) + x.

    This is a simple helper module for residual connections in ConvMixer.

    Args:
        fn: The function/module to apply
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fn(x) + x."""
        return self.fn(x) + x


class ConvMixer(nn.Module):
    """
    ConvMixer architecture for CIFAR-100 (trained from scratch).

    ConvMixer is an extremely simple architecture that:
    1. Uses patch embeddings like ViT
    2. Separates spatial and channel mixing
    3. Uses only standard convolutions (no self-attention)

    Despite its simplicity, it achieves competitive results with more complex
    architectures like ViT and MLP-Mixer.

    Args:
        dim: Hidden dimension (number of channels)
        depth: Number of ConvMixer blocks
        kernel_size: Kernel size for depthwise convolution (spatial mixing)
        patch_size: Patch size for initial embedding (stride of patch conv)
        num_classes: Number of output classes (100 for CIFAR-100)

    Architecture:
        1. Patch Embedding: Conv2d(3, dim, kernel=patch_size, stride=patch_size)
        2. Repeat depth times:
           a. Depthwise Conv (spatial mixing) + Residual
           b. Pointwise Conv (channel mixing)
        3. Global Average Pooling + Linear classifier

    Reference:
        Trockman & Kolter "Patches Are All You Need?" (ICLR 2022)
        https://openreview.net/forum?id=TVHS5Y4dNvM
        https://github.com/locuslab/convmixer

    Example:
        >>> model = ConvMixer(dim=768, depth=32, kernel_size=7, patch_size=2)
        >>> x = torch.randn(2, 3, 32, 32)
        >>> y = model(x)
        >>> assert y.shape == (2, 100)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 9,
        patch_size: int = 7,
        num_classes: int = 100,
    ):
        super().__init__()

        # Patch embedding: project input to dim channels with patch_size stride
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        # ConvMixer blocks: alternate between spatial and channel mixing
        blocks = []
        for _ in range(depth):
            # Spatial mixing: depthwise convolution with residual
            blocks.append(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size=kernel_size,
                            groups=dim,  # Depthwise
                            padding=kernel_size // 2,
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                )
            )

            # Channel mixing: pointwise convolution (1x1)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
            )

        self.blocks = nn.Sequential(*blocks)

        # Classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using standard techniques."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 3, height, width]

        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def convmixer_768_32(num_classes: int = 100) -> ConvMixer:
    """
    ConvMixer-768/32 for CIFAR-100 (trained from scratch).

    Medium-sized ConvMixer variant from the original paper.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)

    Returns:
        ConvMixer-768/32 model instance

    Model Statistics:
        - Parameters: ~21M
        - Hidden dim: 768
        - Depth: 32 blocks
        - Kernel size: 7
        - Patch size: 2 (32x32 → 16x16 patches)
        - Expected F1: 0.83-0.85
        - Training time: ~6-8 hours (600 epochs)

    Paper Results (ImageNet, from scratch):
        - Top-1 Accuracy: 80.2%

    Reference:
        https://github.com/locuslab/convmixer (official implementation)
    """
    return ConvMixer(
        dim=768,
        depth=32,
        kernel_size=7,
        patch_size=2,  # Adapted for CIFAR-100's 32x32 input
        num_classes=num_classes,
    )


def convmixer_1536_20(num_classes: int = 100) -> ConvMixer:
    """
    ConvMixer-1536/20 for CIFAR-100 (trained from scratch).

    Largest ConvMixer variant from the original paper.
    Wider but shallower than 768/32.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)

    Returns:
        ConvMixer-1536/20 model instance

    Model Statistics:
        - Parameters: ~52M
        - Hidden dim: 1536
        - Depth: 20 blocks
        - Kernel size: 9
        - Patch size: 2 (32x32 → 16x16 patches)
        - Expected F1: 0.84-0.86
        - Training time: ~8-10 hours (600 epochs)

    Paper Results (ImageNet, from scratch):
        - Top-1 Accuracy: 81.4%

    Reference:
        https://github.com/locuslab/convmixer (official implementation)
    """
    return ConvMixer(
        dim=1536,
        depth=20,
        kernel_size=9,
        patch_size=2,  # Adapted for CIFAR-100's 32x32 input
        num_classes=num_classes,
    )


def convmixer_1024_20(num_classes: int = 100) -> ConvMixer:
    """
    ConvMixer-1024/20 for CIFAR-100 (trained from scratch).

    Balanced variant with good capacity and efficiency.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)

    Returns:
        ConvMixer-1024/20 model instance

    Model Statistics:
        - Parameters: ~24M
        - Hidden dim: 1024
        - Depth: 20 blocks
        - Kernel size: 9
        - Patch size: 4 (32x32 → 8x8 patches)
        - Expected F1: 0.83-0.85
        - Training time: ~7-9 hours (600 epochs)

    Note:
        This variant uses patch_size=4 for a different patch granularity,
        potentially capturing different levels of detail.

    Reference:
        Adapted from https://github.com/locuslab/convmixer
    """
    return ConvMixer(
        dim=1024,
        depth=20,
        kernel_size=9,
        patch_size=4,  # Larger patch size for different granularity
        num_classes=num_classes,
    )
