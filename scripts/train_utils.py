import os
from typing import Literal, Optional, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

class_names_2_idx: dict[str, int] = {}


# ============================================================================
# Category-Adaptive Augmentation Constants
# ============================================================================
# Based on ResNet-50 training results analysis (F1=0.81)
# Strategy: Different augmentation for different category characteristics

# Group 1: Detail-sensitive classes (Human + Small Animals)
# These classes suffered severe performance drop (-0.14 avg) with CutMix
# Strategy: Use Mixup ONLY (alpha=0.4) to preserve fine-grained features
detail_sensitive_classes = {
    # Human classes (face/clothing details crucial)
    # baby
    # boy
    # girl
    # man
    # woman
    # Small animals (texture details easily destroyed by CutMix)
    # beaver
    # mouse
    # otter
    # possum
    # shrew
}

# Group 2: Local-feature classes (Mechanical + Plants)
# These classes showed excellent performance (0.90+) with CutMix
# Strategy: Prefer CutMix (80% CutMix, 20% Mixup) to enhance local features
local_feature_classes = {
    # Mechanical (local features like wheels, body parts)
    # bicycle
    # bus
    # motorcycle
    # pickup_truck
    # tank
    # tractor
    # train
    # Plants (local texture/shape features)
    # maple_tree
    # oak_tree
    # orchid
    # palm_tree
    # pine_tree
    # sunflower
    # tulip
    # willow_tree
}

# Group 3: Mixed-strategy classes (All others)
# Default strategy: 30% Mixup, 70% CutMix (balanced approach)
# No explicit set needed, used as fallback
# ============================================================================


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch of data.

    Mixup is a data augmentation technique that generates virtual training examples
    by mixing pairs of examples and their labels. It has been shown to improve
    generalization and reduce overfitting, especially on small datasets like CIFAR.

    Formula:
        lambda ~ Beta(alpha, alpha)
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    Args:
        x: Input batch of images [batch_size, C, H, W]
        y: Input batch of labels [batch_size]
        alpha: Mixup hyperparameter (default: 1.0)
               - alpha = 0: No mixup (returns original data)
               - alpha = 1.0: Uniform mixing (recommended for CIFAR)
               - alpha > 1.0: More aggressive mixing
        device: Device to perform operations on

    Returns:
        mixed_x: Mixed images [batch_size, C, H, W]
        y_a: First set of labels [batch_size]
        y_b: Second set of labels [batch_size]
        lam: Mixing coefficient (scalar)

    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
        https://arxiv.org/abs/1710.09412

    Usage:
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
        outputs = model(mixed_x)
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    """
    if alpha > 0:
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
    else:
        # No mixup
        lam = 1.0

    batch_size = x.size(0)

    # Generate random permutation of indices
    # Use x.device to ensure index is on the same device as input tensor
    index = torch.randperm(batch_size, device=x.device)

    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Get corresponding labels
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to a batch of data.

    CutMix is a data augmentation technique that cuts and pastes patches between
    training images. Unlike Mixup which blends entire images, CutMix preserves
    local features by replacing a rectangular region of one image with a patch
    from another. This is especially beneficial for small images like CIFAR where
    preserving local features is crucial.

    Formula:
        lambda ~ Beta(alpha, alpha)
        Generate random bbox with area proportional to (1-lambda)
        x_mixed = x with bbox region replaced by x_j[bbox]
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    Args:
        x: Input batch of images [batch_size, C, H, W]
        y: Input batch of labels [batch_size]
        alpha: CutMix hyperparameter (default: 1.0)
               - alpha = 0: No CutMix (returns original data)
               - alpha = 1.0: Uniform mixing (recommended for CIFAR-100)
               - Higher alpha = larger cut regions
        device: Device to perform operations on

    Returns:
        mixed_x: Mixed images [batch_size, C, H, W]
        y_a: First set of labels [batch_size]
        y_b: Second set of labels [batch_size]
        lam: Mixing coefficient based on actual bbox area (scalar)

    Reference:
        Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers
        with Localizable Features" (ICCV 2019)
        https://arxiv.org/abs/1905.04899

    Key Advantages over Mixup for CIFAR:
        - Preserves local features (no blurring)
        - Forces model to learn from local regions
        - Better for fine-grained classification (seal, otter, boy, girl, etc.)

    Usage:
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        outputs = model(mixed_x)
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    """
    if alpha > 0:
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
    else:
        # No CutMix
        lam = 1.0

    batch_size = x.size(0)

    # Generate random permutation of indices
    # Use x.device to ensure index is on the same device as input tensor
    index = torch.randperm(batch_size, device=x.device)

    # Generate random bounding box
    W, H = x.size(2), x.size(3)  # Width and Height of images

    # Calculate cut size based on lambda
    # cut_rat represents the ratio of the cut region
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of bbox center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate bbox coordinates (x1, y1, x2, y2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix: paste bbox from shuffled images
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda based on actual bbox area (in case of clipping at boundaries)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # Get corresponding labels
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def adaptive_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 0.6,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply category-adaptive augmentation (Mixup or CutMix) based on batch label distribution.

    This function dynamically selects augmentation strategy based on the characteristics
    of classes present in the current batch. Different categories benefit from different
    augmentation techniques:

    **Strategy**:
    1. **Detail-sensitive classes** (human, small animals):
       - Use Mixup ONLY with alpha=0.4 (stronger mixing)
       - Preserves fine-grained features like faces, fur textures
       - Prevents CutMix from destroying crucial details

    2. **Local-feature classes** (mechanical, plants):
       - Prefer CutMix (80%) with standard alpha
       - Enhances learning of local features (wheels, leaves, etc.)
       - Allow 20% Mixup for diversity

    3. **Mixed-strategy classes** (all others):
       - Default strategy: 30% Mixup, 70% CutMix
       - Balanced approach for general categories

    **Decision Logic**:
    - Count classes in each group within the batch
    - Choose strategy based on majority group
    - If no clear majority, use default mixed strategy

    **Motivation**:
    Based on ResNet-50 training results (F1=0.81), we observed:
    - Human classes (girl, boy, woman) dropped -0.14 F1 with heavy CutMix
    - Mechanical classes (pickup_truck, motorcycle) achieved 0.90+ F1 with CutMix
    - This adaptive approach aims to recover performance on detail-sensitive classes
      while maintaining excellent performance on local-feature classes

    Args:
        x: Input batch of images [batch_size, C, H, W]
        y: Input batch of labels [batch_size]
        mixup_alpha: Mixup alpha parameter for mixed-strategy/local-feature classes
        cutmix_alpha: CutMix alpha parameter for local-feature/mixed-strategy classes
        device: Device to perform operations on

    Returns:
        Tuple of (mixed_x, targets_a, targets_b, lambda):
            - mixed_x: Augmented images [batch_size, C, H, W]
            - targets_a: First set of labels [batch_size]
            - targets_b: Second set of labels (from shuffled batch) [batch_size]
            - lam: Mixing coefficient (scalar)

    Example:
        >>> # Batch with majority human classes -> Uses Mixup only
        >>> mixed_x, y_a, y_b, lam = adaptive_augmentation(
        ...     x, y, mixup_alpha=0.2, cutmix_alpha=0.6, device='cuda'
        ... )
        >>> loss = lam * criterion(model(mixed_x), y_a) + (1-lam) * criterion(model(mixed_x), y_b)

    Note:
        This function analyzes the ENTIRE batch to make a single decision.
        For more fine-grained control (per-sample), consider implementing
        class-specific data loaders or weighted sampling strategies.
    """
    # Analyze batch label distribution
    batch_labels = y.cpu().numpy()

    # Count classes in each group
    global detail_sensitive_classes, local_feature_classes
    detail_count = sum(1 for label in batch_labels if label in detail_sensitive_classes)
    local_count = sum(1 for label in batch_labels if label in local_feature_classes)
    mixed_count = len(batch_labels) - detail_count - local_count

    # Decision: Choose augmentation based on majority group
    if detail_count > local_count and detail_count > mixed_count:
        # Majority are detail-sensitive: Use Mixup only (stronger alpha=0.4)
        # This preserves fine-grained features crucial for human/small-animal classification
        return mixup_data(x, y, alpha=0.4, device=device)

    elif local_count > detail_count and local_count > mixed_count:
        # Majority are local-feature: Prefer CutMix (80% CutMix, 20% Mixup)
        # CutMix enhances local feature learning for mechanical/plant categories
        if np.random.rand() < 0.8:
            return cutmix_data(x, y, alpha=cutmix_alpha, device=device)
        else:
            return mixup_data(x, y, alpha=mixup_alpha, device=device)

    else:
        # Mixed or equal distribution: Default strategy (30% Mixup, 70% CutMix)
        # This is the baseline strategy used in previous training
        if np.random.rand() < 0.3:
            return mixup_data(x, y, alpha=mixup_alpha, device=device)
        else:
            return cutmix_data(x, y, alpha=cutmix_alpha, device=device)


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute loss for Mixup augmented data.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss, FocalLoss)
        pred: Model predictions [batch_size, num_classes]
        y_a: First set of labels [batch_size]
        y_b: Second set of labels [batch_size]
        lam: Mixing coefficient (scalar)

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def self_distillation_loss(
    all_logits: list[torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Self-distillation loss (Be Your Own Teacher) - Fixed version.

    Combines two types of losses:
    1. Cross-entropy loss for final classifier only (hard labels)
    2. KL divergence loss (shallow classifiers learn from deep classifier)

    Args:
        all_logits: List of logits from all classifiers [logits1, logits2, logits3, logits4]
                    where logits4 is the deepest (teacher)
        labels: True labels [batch_size]
        temperature: Temperature for knowledge distillation (default: 3.0, lowered from 4.0)
        alpha: Weight for soft labels (default: 0.7, lowered from 0.9 for stability)
               loss = alpha * KL + (1-alpha) * CE

    Returns:
        Total self-distillation loss

    Reference:
        Zhang et al. "Be Your Own Teacher" (2019)
        https://arxiv.org/abs/1905.08094

    Note:
        Modified from paper to avoid NaN:
        - Only final classifier uses CE loss (not all 4)
        - Lower temperature (3.0 vs 4.0) for stability
        - Lower alpha (0.7 vs 0.9) to balance hard/soft labels
    """
    num_classifiers = len(all_logits)
    teacher_logits = all_logits[-1]  # Deepest classifier as teacher

    # Loss 1: Cross-entropy with hard labels (ONLY final classifier to avoid 4x amplification)
    ce_loss = F.cross_entropy(teacher_logits, labels)
    total_loss = (1 - alpha) * ce_loss

    # Loss 2: KL divergence (shallow learn from deep)
    kl_total = 0.0
    for i in range(num_classifiers - 1):  # Exclude teacher itself
        student_logits = all_logits[i]

        # Soft targets from teacher (with detach to avoid gradient issues)
        with torch.no_grad():
            soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

        soft_student = F.log_softmax(student_logits / temperature, dim=1)

        # KL divergence with numerical stability
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            temperature**2
        )

        kl_total += kl_loss

    # Average KL loss across student classifiers (3 classifiers)
    total_loss = total_loss + (alpha / (num_classifiers - 1)) * kl_total

    return total_loss


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model weights and buffers.

    Maintains a moving average of model parameters AND buffers (e.g., BatchNorm statistics)
    which often provides better generalization performance than the final trained weights.

    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999). Higher = slower update.
            Common values: 0.999, 0.9995, 0.9999
        device: Device to store EMA model

    Usage::

        ema = ModelEMA(model, decay=0.9999)
        # During training, after optimizer.step():
        ema.update(model)
        # For evaluation/validation:
        ema.apply_shadow()  # Use EMA weights and buffers
        evaluate(model, ...)
        ema.restore()  # Restore original weights and buffers

    Note:
        This implementation tracks both parameters (weights) and buffers (e.g., BatchNorm's
        running_mean and running_var), which is critical for models with BatchNorm layers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "cpu"):
        self.model = model
        self.decay = decay
        self.device = device

        # Create copies for EMA parameters and buffers
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters (trainable weights)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)

        # Initialize shadow buffers (e.g., BatchNorm running_mean/running_var)
        for name, buffer in model.named_buffers():
            self.shadow[name] = buffer.data.clone().to(device)

    def update(self, model: nn.Module):
        """
        Update EMA parameters and buffers.

        Args:
            model: Current model with updated parameters and buffers
        """
        # Update trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow"
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

        # Update buffers (e.g., BatchNorm statistics)
        for name, buffer in model.named_buffers():
            assert name in self.shadow, f"Buffer {name} not in shadow"
            new_average = (
                self.decay * self.shadow[name] + (1.0 - self.decay) * buffer.data
            )
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters and buffers to model (for evaluation)."""
        # Backup and apply shadow parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow"
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

        # Backup and apply shadow buffers
        for name, buffer in self.model.named_buffers():
            assert name in self.shadow, f"Buffer {name} not in shadow"
            self.backup[name] = buffer.data.clone()
            buffer.data = self.shadow[name].clone()

    def restore(self):
        """Restore original model parameters and buffers."""
        # Restore parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"Parameter {name} not in backup"
                param.data = self.backup[name].clone()

        # Restore buffers
        for name, buffer in self.model.named_buffers():
            assert name in self.backup, f"Buffer {name} not in backup"
            buffer.data = self.backup[name].clone()

        self.backup = {}


class AlbumentationsTransform:
    """Wrapper to use Albumentations with PyTorch DataLoader."""

    def __init__(self, transform: A.Compose):
        """
        Initialize the wrapper.

        Args:
            transform: Albumentations Compose object
        """
        self.transform = transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply Albumentations transform to PIL Image.

        Args:
            img: PIL Image

        Returns:
            Transformed image as torch.Tensor
        """
        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Apply Albumentations transforms
        augmented = self.transform(image=img_np)
        img_transformed = augmented["image"]

        return img_transformed


def get_train_transforms(
    dataset_type: Literal["cifar10", "cifar100"] = "cifar100",
    augmentation_strength: Literal[
        "light", "medium", "strong", "randaugment"
    ] = "light",
    use_cutmix: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
) -> AlbumentationsTransform:
    """
    Get training transforms with online augmentation.

    Args:
        dataset_type: Type of dataset ("cifar10" or "cifar100")
        augmentation_strength: Augmentation strategy. Options:
            - "light": Light traditional augmentation
            - "medium": Medium traditional augmentation
            - "strong": Strong traditional augmentation
            - "randaugment": Pure RandAugment (replaces traditional aug)
        use_cutmix: Whether CutMix is used (affects CoarseDropout in traditional aug)
        randaugment_n: Number of RandAugment operations (default: 2, used when aug_strength="randaugment")
        randaugment_m: Magnitude of RandAugment (0-10, default: 9, used when aug_strength="randaugment")

    Returns:
        AlbumentationsTransform wrapper with augmentation pipeline
    """
    # Get dataset-specific normalization statistics
    if dataset_type == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_type == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    # Build augmentation pipeline based on strength
    if augmentation_strength == "randaugment":
        # Pure RandAugment mode (no stacking with traditional augmentation)
        # This is the correct way to use RandAugment per the original paper
        # F1=0.8131 achieved with N=2, M=9
        from scripts.data_augmentation import RandAugment

        augmentation_pipeline = A.Compose(  # type: ignore[arg-type]
            [
                RandAugment(n=randaugment_n, m=randaugment_m),
                A.HorizontalFlip(p=0.5),  # Basic geometric transform
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif augmentation_strength == "light":
        # Light augmentations - suitable for CIFAR and Focal Loss
        augmentation_pipeline = A.Compose(  # type: ignore[arg-type]
            [
                A.Rotate(limit=15, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.8,
                    border_mode=0,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif augmentation_strength == "medium":
        # Medium augmentations (traditional)
        augmentation_pipeline = A.Compose(  # type: ignore[arg-type]
            [
                A.Rotate(limit=20, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.8,
                    border_mode=0,
                ),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                        A.MedianBlur(blur_limit=5, p=0.3),
                    ],
                    p=0.4,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ]
            + (
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 1),
                        hole_height_range=(4, 8),
                        hole_width_range=(4, 8),
                        p=0.3,
                    ),
                ]
                if not use_cutmix
                else []
            )
            + [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif augmentation_strength == "strong":
        # Strong augmentations (traditional, not recommended for 32x32 images)
        augmentation_pipeline = A.Compose(  # type: ignore[arg-type]
            [
                A.Rotate(limit=25, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.8,
                    border_mode=0,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                        A.GridDistortion(p=0.3),
                        A.OpticalDistortion(distort_limit=0.3, p=0.3),
                    ],
                    p=0.4,
                ),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.9
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=0.5,
                        ),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5
                        ),
                        A.ChannelShuffle(p=0.3),
                        A.RandomToneCurve(p=0.3),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                        A.MotionBlur(blur_limit=7, p=0.4),
                        A.MedianBlur(blur_limit=5, p=0.3),
                    ],
                    p=0.5,
                ),
                A.GaussNoise(p=0.4),
                A.OneOf(
                    [
                        A.Sharpen(p=0.3),
                        A.Emboss(p=0.3),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.6
                ),
            ]
            + (
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(2, 8),
                        hole_width_range=(2, 8),
                        p=0.5,
                    ),
                ]
                if not use_cutmix
                else []
            )
            + [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(
            f"Unknown augmentation_strength: {augmentation_strength}. "
            f"Choose from 'light', 'medium', 'strong'."
        )

    return AlbumentationsTransform(augmentation_pipeline)


def load_transforms(dataset_type: str = "cifar100"):
    """
    Load the data transformations with correct normalization statistics.

    Args:
        dataset_type: Type of dataset ("cifar10" or "cifar100")

    Returns:
        Composed transforms with appropriate normalization
    """
    # Use dataset-specific normalization statistics (computed from training set)
    if dataset_type == "cifar10":
        # CIFAR-10 statistics
        # Ref: https://github.com/kuangliu/pytorch-cifar
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_type == "cifar100":
        # CIFAR-100 statistics
        # Ref: https://github.com/weiaicunzai/pytorch-cifar100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        # Fallback to generic normalization
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_data(
    data_dir: str,
    batch_size: int,
    dataset_type: Literal["cifar10", "cifar100"] = "cifar100",
    manual_seed: int = 42,
    use_online_aug: bool = True,
    augmentation_strength: Literal[
        "light", "medium", "strong", "randaugment"
    ] = "light",
    use_cutmix: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
):
    """
    Load the data from the data directory and split it into training and validation sets.

    Supports two augmentation strategies:
    1. ONLINE augmentation: Apply transforms dynamically during training (different every epoch)
    2. OFFLINE augmentation: Load pre-generated augmented data (larger dataset, static)

    Args:
        data_dir: The directory to load the data from
                  - For online aug: use raw data directory (e.g., data/raw/train)
                  - For offline aug: use augmented data directory (e.g., data/augmented/train)
        batch_size: The batch size to use for the data loaders
        dataset_type: Type of dataset ("cifar10" or "cifar100") for proper normalization
        manual_seed: Random seed for reproducible train/val split
        use_online_aug: Whether to use online augmentation (default: True)
                        - True: Apply augmentations dynamically (recommended for memory efficiency)
                        - False: Use pre-generated augmented data (more training samples)
        augmentation_strength: Augmentation strength for training ("light", "medium", "strong")
                               Only used when use_online_aug=True

    Returns:
        train_loader: The training data loader
        val_loader: The validation data loader (always without augmentation)
    """
    # Determine transforms based on augmentation strategy
    if use_online_aug:
        # ONLINE augmentation: apply transforms dynamically
        train_transforms = get_train_transforms(
            dataset_type=dataset_type,
            augmentation_strength=augmentation_strength,
            use_cutmix=use_cutmix,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
        )
    else:
        # OFFLINE augmentation: data is already augmented, just normalize
        train_transforms = load_transforms(dataset_type=dataset_type)

    # Validation always uses standard transforms (no augmentation)
    val_transforms = load_transforms(dataset_type=dataset_type)

    # Load the train dataset
    # CRITICAL FIX: Ensure we're loading from the correct directory
    # For online aug: data_dir should be "data/raw/train" (40000 images after split)
    # For offline aug: data_dir should be "data/augmented/train" (40000 * aug_count images)
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Load the validation dataset
    # CRITICAL FIX: Construct validation path correctly
    # Extract base data directory (e.g., "data") and construct validation path
    import os

    if "augmented" in data_dir:
        # Offline aug: data_dir = "data/augmented/train" -> val_dir = "data/raw/val"
        base_dir = os.path.dirname(os.path.dirname(data_dir))  # "data"
        val_dir = os.path.join(base_dir, "raw", "val")
    else:
        # Online aug: data_dir = "data/raw/train" -> val_dir = "data/raw/val"
        base_dir = os.path.dirname(os.path.dirname(data_dir))  # "data"
        val_dir = os.path.join(base_dir, "raw", "val")

    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    # Create data loaders for training and validation
    # Optimized DataLoader for better GPU utilization
    # Training has CPU-intensive operations (online augmentation, Mixup/CutMix)
    # More workers and prefetch help keep GPU busy
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,
    )
    # Validation has less CPU overhead (no augmentation, no backward pass)
    # Moderate increase in workers is sufficient
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Print dataset summary
    print(f"Dataset loaded from: {data_dir}")
    print(
        f"Augmentation strategy: {'ONLINE (dynamic)' if use_online_aug else 'OFFLINE (pre-generated)'}"
    )
    print(f"Total images: {len(train_dataset) + len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # build class_names_2_idx
    global class_names_2_idx
    class_names_2_idx = {name: idx for idx, name in enumerate(train_dataset.classes)}
    print(f"COUNT(class_names_2_idx): {len(class_names_2_idx)}")
    print(f"Class names to indices: {class_names_2_idx}")

    # build detail_sensitive_classes
    global detail_sensitive_classes
    detail_sensitive_classes = {
        # Human classes (face/clothing details crucial)
        class_names_2_idx["baby"],
        class_names_2_idx["boy"],
        class_names_2_idx["girl"],
        class_names_2_idx["man"],
        class_names_2_idx["woman"],
        # Small animals (texture details easily destroyed by CutMix)
        class_names_2_idx["beaver"],
        class_names_2_idx["mouse"],
        class_names_2_idx["otter"],
        class_names_2_idx["possum"],
        class_names_2_idx["shrew"],
    }

    # build local_feature_classes
    global local_feature_classes
    local_feature_classes = {
        # Mechanical (local features like wheels, body parts)
        class_names_2_idx["bicycle"],
        class_names_2_idx["bus"],
        class_names_2_idx["motorcycle"],
        class_names_2_idx["pickup_truck"],
        class_names_2_idx["tank"],
        class_names_2_idx["tractor"],
        class_names_2_idx["train"],
        # Plants (local texture/shape features)
        class_names_2_idx["maple_tree"],
        class_names_2_idx["oak_tree"],
        class_names_2_idx["orchid"],
        class_names_2_idx["palm_tree"],
        class_names_2_idx["pine_tree"],
        class_names_2_idx["sunflower"],
        class_names_2_idx["tulip"],
        class_names_2_idx["willow_tree"],
    }

    return train_loader, val_loader


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing is a regularization technique that prevents the model from
    becoming over-confident by smoothing the target distribution.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits) of shape [batch_size, num_classes]
            target: Ground truth labels of shape [batch_size]

        Returns:
            Smoothed cross-entropy loss
        """
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            # Create smoothed target distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.

    Focal Loss down-weights easy examples and focuses on hard negatives,
    which is particularly useful for datasets with class imbalance or
    when some samples are much harder to classify than others.

    Formula: FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.

    Args:
        alpha: Weighting factor for class imbalance (optional).
               Can be a scalar or a tensor of shape [num_classes].
        gamma: Focusing parameter (default: 2.0). Higher gamma reduces
               loss for well-classified examples. Recommended: 1.0-3.0.
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]

        Returns:
            Focal loss value
        """
        # Compute cross entropy loss (without reduction)
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")

        # Compute p_t: the probability of the true class
        p_t = torch.exp(-ce_loss)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to CE loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def define_loss_and_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adam",
    scheduler_type: Literal["plateau", "cosine", "step", "onecycle"] = "plateau",
    label_smoothing: float = 0.0,
    num_epochs: int = 100,
    steps_per_epoch: Optional[int] = None,
    warmup_epochs: int = 5,
    loss_type: Literal["ce", "focal"] = "ce",
    focal_gamma: float = 2.0,
    use_class_weights: bool = False,
    num_classes: int = 100,
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """
    Define the loss function, optimizer, and learning rate scheduler.

    This function provides multiple optimizer and scheduler options for improved training.

    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        optimizer_type: Type of optimizer. Options:
            - "adam": Adam optimizer (original default)
            - "adamw": AdamW optimizer (Adam with decoupled weight decay)
            - "sgd": SGD with momentum (often better for ResNets)
        scheduler_type: Type of learning rate scheduler. Options:
            - "plateau": ReduceLROnPlateau (original default, reduces lr when validation loss stops improving)
            - "cosine": CosineAnnealingLR (smooth lr decay following cosine curve)
            - "step": StepLR (lr decay at fixed intervals)
            - "onecycle": OneCycleLR (fast convergence with cyclical lr)
        label_smoothing: Label smoothing factor (0.0-1.0). 0.0 = no smoothing.
            Recommended: 0.1 for CIFAR-100 to prevent overconfidence.
            Note: Not compatible with focal loss.
        num_epochs: Total number of training epochs (required for some schedulers)
        steps_per_epoch: Number of steps per epoch (required for OneCycleLR)
        warmup_epochs: Number of warmup epochs for cosine scheduler (default: 5)
        loss_type: Type of loss function. Options:
            - "ce": CrossEntropyLoss (standard, works with label smoothing)
            - "focal": Focal Loss (addresses class imbalance and hard examples)
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0).
            Higher values focus more on hard examples. Recommended range: 1.0-3.0.

    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The learning rate scheduler

    Raises:
        ValueError: If optimizer_type or scheduler_type is not recognized
    """
    # Compute class weights for hard examples (based on empirical difficulty)
    class_weights = None
    if use_class_weights:
        # Hard-coded weights for difficult CIFAR-100 classes
        # Base weight = 1.0, increase for consistently low-performing classes
        weights = torch.ones(num_classes)

        global class_names_2_idx
        # Define hard class indices (0-indexed, alphabetically sorted CIFAR-100 classes)
        # IMPORTANT: Indices are based on alphabetically sorted class names!
        # Updated based on latest training_metrics.txt (Mixup-CE-OptArgs run)
        #
        # Very hard classes (F1 < 0.50): weight=3.0
        #   girl(F1=0.42), seal(F1=0.43), otter(F1=0.45),
        #   shrew(F1=0.48), lizard(F1=0.50), boy(F1=0.50)
        very_hard_classes = [
            class_names_2_idx["girl"],
            class_names_2_idx["seal"],
            class_names_2_idx["otter"],
            class_names_2_idx["shrew"],
            class_names_2_idx["lizard"],
            class_names_2_idx["boy"],
        ]  # weight = 3.0

        # Hard classes (0.50 <= F1 < 0.60): weight=2.0
        #   mouse(F1=0.52), squirrel(F1=0.52), lobster(F1=0.53),
        #   man(F1=0.54), woman(F1=0.54), rabbit(F1=0.54),
        #   bear(F1=0.55), beaver(F1=0.55), possum(F1=0.56)
        hard_classes = [
            class_names_2_idx["mouse"],
            class_names_2_idx["squirrel"],
            class_names_2_idx["lobster"],
            class_names_2_idx["man"],
            class_names_2_idx["woman"],
            class_names_2_idx["rabbit"],
            class_names_2_idx["bear"],
            class_names_2_idx["beaver"],
            class_names_2_idx["possum"],
        ]  # weight = 2.0

        for idx in very_hard_classes:
            if idx < num_classes:
                weights[idx] = 3.0
        for idx in hard_classes:
            if idx < num_classes:
                weights[idx] = 2.0

        class_weights = weights.to(next(model.parameters()).device)
        print(
            f"Using class weights (for training / validation):\n\tvery_hard={very_hard_classes[:3]}... (3.0x), hard={hard_classes[:3]}... (2.0x)"
        )

    # Define loss function
    if loss_type == "focal":
        # Focal Loss for addressing class imbalance and hard examples
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        if label_smoothing > 0.0:
            print(
                f"Warning: Label smoothing ({label_smoothing}) is not compatible "
                f"with Focal Loss. Using Focal Loss without label smoothing."
            )
            label_smoothing = 0.0
            print("    Label smoothing was set to 0.0!")
    elif loss_type == "ce":
        # Standard CrossEntropyLoss with optional label smoothing
        if label_smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Available options: 'ce', 'focal'"
        )

    # Define optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. "
            f"Available options: 'adam', 'adamw', 'sgd'"
        )

    # Define learning rate scheduler
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5,
        )
    elif scheduler_type == "cosine":
        # Cosine Annealing with warmup
        if warmup_epochs > 0:
            # Warmup: linear increase from lr/10 to lr
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at lr * 0.1
                end_factor=1.0,  # End at lr * 1.0
                total_iters=warmup_epochs,
            )
            # Main scheduler: cosine annealing
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr * 0.01
            )
            # Combine warmup and cosine
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            # No warmup, just cosine
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=lr * 0.01
            )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR scheduler")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy="cos",
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=1e4,  # min_lr = initial_lr/1e4
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Available options: 'plateau', 'cosine', 'step', 'onecycle'"
        )

    return criterion, optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
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
    use_self_distill: bool = False,
    distill_temperature: float = 4.0,
    distill_alpha: float = 0.9,
) -> Tuple[float, float]:
    """
    Train the model for one epoch with optional mixed precision training, gradient clipping, Mixup, and CutMix.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Optional LR scheduler (for OneCycleLR that updates per batch)
        use_amp: Whether to use Automatic Mixed Precision training
        max_grad_norm: Maximum gradient norm for gradient clipping. None = no clipping.
            Recommended: 1.0 for stable training
        ema: Optional EMA object to update after each batch
        mixup_alpha: Mixup alpha parameter (default: 0.0 = no mixup)
            - 0.0: Disabled (normal training)
            - 1.0: Uniform mixing (recommended for CIFAR)
            - 0.2-0.4: Light mixing
        cutmix_alpha: CutMix alpha parameter (default: 0.0 = no cutmix)
            - 0.0: Disabled
            - 1.0: Recommended for CIFAR-100
        use_cutmix: Whether to use CutMix

    Returns:
        Tuple of (average loss, accuracy percentage) for the epoch

    Note:
        If both CutMix and Mixup are enabled, will randomly choose one per batch (40% Mixup, 60% CutMix).
        This CutMix-prioritized approach leverages strong regularization for fine-grained categories
        while maintaining Mixup's benefits for small animal classification.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.grad_scaler.GradScaler(device=DEVICE_TYPE) if use_amp else None

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply data augmentation (category-adaptive or fixed strategy)
        if use_cutmix and cutmix_alpha > 0 and mixup_alpha > 0:
            # Use category-adaptive augmentation based on batch label distribution
            # This dynamically selects Mixup/CutMix based on class characteristics:
            # - Detail-sensitive (human, small animals): Mixup only (alpha=0.4)
            # - Local-feature (mechanical, plants): 80% CutMix, 20% Mixup
            # - Mixed-strategy (others): 30% Mixup, 70% CutMix
            inputs, targets_a, targets_b, lam = adaptive_augmentation(
                inputs,
                labels,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                device=device,
            )
        elif use_cutmix and cutmix_alpha > 0:
            # Apply CutMix augmentation only
            inputs, targets_a, targets_b, lam = cutmix_data(
                inputs, labels, alpha=cutmix_alpha, device=device
            )
        elif mixup_alpha > 0:
            # Apply Mixup augmentation only
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, alpha=mixup_alpha, device=device
            )
        else:
            # No augmentation, use original labels
            targets_a, targets_b, lam = labels, labels, 1.0

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if use_amp:
            with torch.amp.autocast_mode.autocast(device_type=DEVICE_TYPE):
                # Self-distillation: get all classifier outputs
                if use_self_distill:
                    all_logits = model(inputs, return_all=True)  # type: ignore
                    outputs = all_logits[-1]  # Final classifier for accuracy

                    # Compute self-distillation loss
                    if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
                        # Mixup/CutMix: apply to all classifiers
                        loss = lam * self_distillation_loss(
                            all_logits, targets_a, distill_temperature, distill_alpha
                        ) + (1 - lam) * self_distillation_loss(
                            all_logits, targets_b, distill_temperature, distill_alpha
                        )
                    else:
                        loss = self_distillation_loss(
                            all_logits, labels, distill_temperature, distill_alpha
                        )
                else:
                    # Normal training
                    outputs = model(inputs)
                    if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
                        loss = mixup_criterion(
                            criterion, outputs, targets_a, targets_b, lam
                        )
                    else:
                        loss = criterion(outputs, labels)
        else:
            # Self-distillation: get all classifier outputs
            if use_self_distill:
                all_logits = model(inputs, return_all=True)  # type: ignore
                outputs = all_logits[-1]  # Final classifier for accuracy

                # Compute self-distillation loss
                if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
                    loss = lam * self_distillation_loss(
                        all_logits, targets_a, distill_temperature, distill_alpha
                    ) + (1 - lam) * self_distillation_loss(
                        all_logits, targets_b, distill_temperature, distill_alpha
                    )
                else:
                    loss = self_distillation_loss(
                        all_logits, labels, distill_temperature, distill_alpha
                    )
            else:
                # Normal training
                outputs = model(inputs)
                if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
                    loss = mixup_criterion(
                        criterion, outputs, targets_a, targets_b, lam
                    )
                else:
                    loss = criterion(outputs, labels)

        # Backward pass and optimize
        if use_amp and scaler:
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for AMP)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        # Update scheduler if it's OneCycleLR (updates every batch)
        if scheduler is not None and isinstance(
            scheduler, optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        # Update EMA if enabled
        if ema is not None:
            ema.update(model)

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)

        # Compute accuracy (different for Mixup/CutMix)
        if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
            # For Mixup/CutMix, compute weighted accuracy based on both labels
            correct += (
                lam * predicted.eq(targets_a).sum().item()
                + (1 - lam) * predicted.eq(targets_b).sum().item()
            )
        else:
            correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """
    Validate the model on validation/test data.

    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy percentage, macro F1-score)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # for F1 calculation
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # gather predictions and labels for F1 calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    # calculate macro average F1-score
    from sklearn.metrics import f1_score

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return epoch_loss, epoch_acc, f1


def save_checkpoint(state, filename):
    """
    Save model checkpoint

    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint

    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file

    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, "w") as f:
        f.write(metrics)
