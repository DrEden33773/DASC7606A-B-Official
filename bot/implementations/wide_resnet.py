"""
Wide ResNet implementation for CIFAR datasets.

This module implements Wide Residual Networks (Wide ResNet) as described in:
"Wide Residual Networks" (Zagoruyko & Komodakis, BMVC 2016)
https://arxiv.org/abs/1605.07146

Wide ResNet improves upon standard ResNet by using wider layers (more channels)
instead of deeper architectures, which is more effective for CIFAR-sized images.

Key features:
- Dropout placement between BN-ReLU-Conv layers
- Pre-activation structure (BN-ReLU-Conv)
- Wider channels with moderate depth (28 layers)

Typical configurations for CIFAR-100:
- WRN-28-10: depth=28, widen_factor=10, ~36.5M params
- WRN-40-10: depth=40, widen_factor=10, ~55.8M params
- WRN-28-12: depth=28, widen_factor=12, ~52.8M params
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                   - Higher values: More aggressive regularization

    References:
        Huang et al. "Deep Networks with Stochastic Depth" (ECCV 2016)
        https://arxiv.org/abs/1603.09382

    Example:
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> out = drop_path(x)  # During training, 20% chance of returning zeros
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor (same shape as input, scaled by keep_prob during training)
        """
        # During evaluation or if drop_prob=0, return input as-is
        if not self.training or self.drop_prob == 0.0:
            return x

        # Calculate keep probability
        keep_prob = 1.0 - self.drop_prob

        # Generate random binary mask (same as batch dimension)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch_size, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize: 0 or 1

        # Scale by keep_prob to maintain expected value
        output = x.div(keep_prob) * random_tensor

        return output


class WideBasicBlock(nn.Module):
    """
    Wide ResNet basic residual block with pre-activation structure.

    This block uses:
    1. Batch Normalization → ReLU → Conv structure (pre-activation)
    2. Dropout between the two conv layers for regularization
    3. Skip connection with optional 1×1 conv for dimension matching

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (1 or 2)
        dropout_rate: Dropout probability (typically 0.3 for CIFAR)

    Shape:
        - Input: [batch_size, in_channels, H, W]
        - Output: [batch_size, out_channels, H/stride, W/stride]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # First BN-ReLU-Conv block
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # Dropout layer between conv blocks (Wide ResNet specific)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Second BN-ReLU-Conv block
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Stochastic Depth (Drop Path)
        self.drop_path = DropPath(drop_path_rate)

        # Skip connection with 1×1 conv if dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wide basic block.

        Args:
            x: Input tensor [batch_size, in_channels, H, W]

        Returns:
            Output tensor [batch_size, out_channels, H', W']
        """
        # First block: BN → ReLU → Conv → Dropout
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)

        # Second block: BN → ReLU → Conv
        out = self.conv2(F.relu(self.bn2(out)))

        # Apply stochastic depth before adding skip connection
        out = self.drop_path(out)

        # Add skip connection
        out = out + self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """
    Wide Residual Network for CIFAR datasets.

    Architecture:
    - Initial 3×3 conv layer (16 channels)
    - 3 groups of residual blocks (with increasing channels)
    - Global average pooling
    - Fully connected layer

    The network depth is calculated as: depth = 6n + 4
    where n is the number of blocks per group.

    Args:
        depth: Total network depth (e.g., 28, 40). Must satisfy (depth-4)%6==0
        widen_factor: Channel width multiplier (e.g., 10, 12)
        num_classes: Number of output classes (100 for CIFAR-100)
        dropout_rate: Dropout rate in residual blocks (default: 0.3)

    Raises:
        AssertionError: If depth is not valid (must satisfy (depth-4)%6==0)

    Example:
        >>> model = WideResNet(depth=28, widen_factor=10, num_classes=100)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([8, 100])

    References:
        Zagoruyko & Komodakis. "Wide Residual Networks" (BMVC 2016)
        https://arxiv.org/abs/1605.07146
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

        # Validate depth parameter
        assert (depth - 4) % 6 == 0, (
            f"Depth must satisfy (depth-4)%6==0, got depth={depth}"
        )

        # Calculate number of blocks per group
        n_blocks = (depth - 4) // 6

        # Calculate number of channels in each stage
        # Stage 0: 16
        # Stage 1: 16 * widen_factor
        # Stage 2: 32 * widen_factor
        # Stage 3: 64 * widen_factor
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Calculate stochastic depth rates (linear increase from 0 to drop_path_rate)
        # For WRN-28-10: 12 blocks total (4 per group × 3 groups)
        total_blocks = n_blocks * 3
        drop_rates = [
            i * drop_path_rate / (total_blocks - 1) if total_blocks > 1 else 0.0
            for i in range(total_blocks)
        ]

        # Track current number of input channels
        self.in_channels = n_channels[0]

        # Initial convolution layer (CIFAR-specific: 3×3 without stride)
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # Create three groups of residual blocks with linearly increasing drop_path_rate
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

        # Final batch normalization
        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.9)

        # Classifier
        self.fc = nn.Linear(n_channels[3], num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        drop_rates: List[float],
    ) -> nn.Sequential:
        """
        Create a layer consisting of multiple wide basic blocks.

        Args:
            out_channels: Number of output channels for this layer
            num_blocks: Number of blocks in this layer
            dropout_rate: Dropout rate for all blocks
            stride: Stride for the first block (1 or 2)
            drop_rates: List of drop_path rates for each block (linearly increasing)

        Returns:
            Sequential container with all blocks
        """
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
        """
        Initialize model weights using Kaiming (He) initialization.

        This is crucial for training deep networks from scratch:
        - Conv layers: Kaiming normal initialization
        - Linear layers: Kaiming normal initialization
        - BN layers: weight=1, bias=0
        """
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
        """
        Forward pass through Wide ResNet.

        Args:
            x: Input tensor [batch_size, 3, 32, 32]

        Returns:
            Output logits [batch_size, num_classes]
        """
        # Initial convolution
        out = self.conv1(x)

        # Three stages of residual blocks
        out = self.layer1(out)  # [B, 16*k, 32, 32]
        out = self.layer2(out)  # [B, 32*k, 16, 16]
        out = self.layer3(out)  # [B, 64*k, 8, 8]

        # Final BN and ReLU
        out = F.relu(self.bn1(out))

        # Global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Classifier
        out = self.fc(out)

        return out


def wide_resnet28_10(
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    drop_path_rate: float = 0.0,
) -> WideResNet:
    """
    Construct Wide ResNet-28-10 model for CIFAR datasets.

    This is the most commonly used configuration for CIFAR-100, offering
    a good balance between model capacity and training efficiency.

    Architecture:
    - Depth: 28 layers (4 blocks per group × 3 groups × 2 conv/block + 4)
    - Width: 10× base channels (160, 320, 640)
    - Parameters: ~36.5M
    - FLOPs: ~5.2G (for 32×32 input)

    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        drop_path_rate: Stochastic depth drop rate (default: 0.0, recommended: 0.2)

    Returns:
        Wide ResNet-28-10 model instance

    Example:
        >>> model = wide_resnet28_10(num_classes=100, dropout_rate=0.3, drop_path_rate=0.2)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([16, 100])
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
    Construct Wide ResNet-40-10 model for CIFAR datasets.

    Deeper variant with more parameters, suitable for Phase 3 optimization.

    Architecture:
    - Depth: 40 layers (6 blocks per group × 3 groups × 2 conv/block + 4)
    - Width: 10× base channels (160, 320, 640)
    - Parameters: ~55.8M
    - FLOPs: ~8.0G (for 32×32 input)

    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        drop_path_rate: Stochastic depth drop rate (default: 0.0, recommended: 0.3)

    Returns:
        Wide ResNet-40-10 model instance
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
    Construct Wide ResNet-28-12 model for CIFAR datasets.

    Wider variant with more channels per layer.

    Architecture:
    - Depth: 28 layers
    - Width: 12× base channels (192, 384, 768)
    - Parameters: ~52.8M
    - FLOPs: ~7.5G (for 32×32 input)

    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        drop_path_rate: Stochastic depth drop rate (default: 0.0, recommended: 0.2)

    Returns:
        Wide ResNet-28-12 model instance
    """
    return WideResNet(
        depth=28,
        widen_factor=12,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )
