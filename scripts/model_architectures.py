from typing import Literal, Optional, Protocol, Type

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
        # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
        # Fully connected layers: flatten feature maps and classify
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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


def resnet18_cifar(num_classes: int = 10, dropout_rate: float = 0.3) -> ResNetCIFAR:
    """
    Construct ResNet-18 model for CIFAR datasets

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization

    Returns:
        ResNet-18 model optimized for CIFAR
    """
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes, dropout_rate)


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


def create_pretrained_resnet(
    num_classes: int,
    device: str,
    model_type: Literal["resnet18", "resnet34", "resnet50"] = "resnet34",
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create ImageNet pretrained ResNet model adapted for CIFAR-100.

    This function loads a ResNet pretrained on ImageNet and adapts it for CIFAR:
    1. Loads pretrained weights from torchvision
    2. Modifies first conv layer for 32×32 input (instead of 224×224)
    3. Removes max pooling to preserve spatial resolution
    4. Replaces final FC layer for target number of classes
    5. Optionally freezes backbone for feature extraction

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        device: Device to place model on ('cuda' or 'cpu')
        model_type: Type of ResNet architecture. Options:
            - "resnet18": ResNet-18 (11M params)
            - "resnet34": ResNet-34 (21M params, recommended)
            - "resnet50": ResNet-50 (25M params)
        dropout_rate: Dropout rate before final FC layer (default: 0.5)
        freeze_backbone: If True, freeze all layers except FC (for initial training)

    Returns:
        Pretrained ResNet model adapted for CIFAR-100

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> # Phase 1: Train only FC layer
        >>> model = create_pretrained_resnet(100, 'cuda', freeze_backbone=True)
        >>>
        >>> # Phase 2: Fine-tune entire model
        >>> model = create_pretrained_resnet(100, 'cuda', freeze_backbone=False)
    """
    # Load pretrained model from torchvision
    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_type == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'resnet18', 'resnet34', 'resnet50'"
        )

    # Modify first conv layer for CIFAR (32×32 instead of 224×224)
    # Original: 7×7 conv with stride=2 → 112×112
    # Modified: 3×3 conv with stride=1 → 32×32 (preserve resolution)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove max pooling layer (preserve 32×32 resolution)
    # Original: 3×3 max pool with stride=2 → 56×56
    # Modified: Identity (no-op) → 32×32
    model.maxpool = nn.Identity()  # type: ignore[assignment]

    # Replace final FC layer for CIFAR-100
    num_features = model.fc.in_features  # 512 for ResNet18/34, 2048 for ResNet50
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
    )

    # Optionally freeze backbone (for phase 1 training)
    if freeze_backbone:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the final FC layer
        for param in model.fc.parameters():
            param.requires_grad = True

    # Move model to device
    model = model.to(device)

    return model


def create_model(
    num_classes: int,
    device: str,
    model_type: Literal["simple", "resnet18", "resnet34", "resnet50"] = "simple",
    dropout_rate: float = 0.3,
    use_pretrained: bool = False,
):
    """
    Create and initialize the model

    Args:
        num_classes: Number of output classes
        device: Device to place the model on ('cuda' or 'cpu')
        model_type: Type of model architecture to use. Options:
            - "simple": SimpleCNN (original baseline model)
            - "resnet18": ResNet-18 optimized for CIFAR (11M params)
            - "resnet34": ResNet-34 optimized for CIFAR (21M params)
            - "resnet50": ResNet-50 optimized for CIFAR (23.5M params, uses Bottleneck)
        dropout_rate: Dropout rate for models that support it (default: 0.3)
        use_pretrained: If True, use ImageNet pretrained weights (Transfer Learning)

    Returns:
        Model instance moved to the specified device

    Raises:
        ValueError: If model_type is not recognized

    Note:
        ResNet-50 uses Bottleneck blocks (1x1->3x3->1x1 structure) which are more
        parameter-efficient than BasicBlock used in ResNet-18/34. Expected to provide
        +0.02-0.03 F1 improvement over ResNet-34 for CIFAR-100 (from scratch).
    """
    # Use pretrained model if requested
    if use_pretrained:
        if model_type not in ["resnet18", "resnet34", "resnet50"]:
            raise ValueError(
                f"Pretrained models only available for ResNet. "
                f"Got model_type='{model_type}'. Use 'resnet18', 'resnet34', or 'resnet50'."
            )
        # Type narrowing for pretrained model types
        pretrained_model_type: Literal["resnet18", "resnet34", "resnet50"]
        if model_type == "resnet18":
            pretrained_model_type = "resnet18"
        elif model_type == "resnet34":
            pretrained_model_type = "resnet34"
        elif model_type == "resnet50":
            pretrained_model_type = "resnet50"
        else:
            raise ValueError(f"Unexpected model_type: {model_type}")

        model = create_pretrained_resnet(
            num_classes=num_classes,
            device=device,
            model_type=pretrained_model_type,
            dropout_rate=dropout_rate,
            freeze_backbone=False,  # Start with unfrozen for full fine-tuning
        )
    else:
        # Create model from scratch
        if model_type == "simple":
            model = SimpleCNN(num_classes=num_classes)
        elif model_type == "resnet18":
            model = resnet18_cifar(num_classes=num_classes, dropout_rate=dropout_rate)
        elif model_type == "resnet34":
            model = resnet34_cifar(num_classes=num_classes, dropout_rate=dropout_rate)
        elif model_type == "resnet50":
            model = resnet50_cifar(num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available options: 'simple', 'resnet18', 'resnet34', 'resnet50'"
            )

        model = model.to(device)

    return model
