"""
RandAugment implementation for CIFAR datasets.

RandAugment is a simple and effective data augmentation strategy that
randomly selects N augmentation operations from a predefined set and
applies them with a uniform magnitude M.

Reference:
    Cubuk et al. "RandAugment: Practical automated data augmentation
    with a reduced search space" (NeurIPS 2020)
    https://arxiv.org/abs/1909.13719

Key advantages:
- Only 2 hyperparameters: N (number of ops) and M (magnitude)
- Works well across different datasets and models
- Simple to implement and tune

Recommended for CIFAR-100:
- N = 2 (apply 2 random operations per image)
- M = 9 (magnitude on scale 0-10)
"""

from typing import List

import albumentations as A
import numpy as np


def get_randaugment_transforms(
    n_ops: int = 2,
    magnitude: int = 9,
) -> List[A.BasicTransform]:
    """
    Get RandAugment transform operations for CIFAR.

    Returns a list of augmentation operations that RandAugment can sample from.
    Each operation is configured with the specified magnitude.

    Args:
        n_ops: Number of operations to apply per image (default: 2)
        magnitude: Magnitude of augmentations on scale 0-10 (default: 9)
                   Higher values = stronger augmentation

    Returns:
        List of Albumentations transforms

    Note:
        The magnitude is mapped to appropriate ranges for each operation.
        For example, magnitude=9 on scale 0-10 maps to ~0.9 of max range.
    """
    # Map magnitude (0-10) to actual parameter ranges
    # magnitude=0 → minimal, magnitude=10 → maximal
    mag_ratio = magnitude / 10.0

    # Define augmentation operations pool (14 operations similar to original RandAugment)
    # Using Albumentations-compatible transforms
    augmentation_pool: List[A.BasicTransform] = [
        # 1. Contrast adjustment
        A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=(mag_ratio * 0.9, mag_ratio * 0.9),
            p=1.0,
        ),
        # 2. Brightness adjustment
        A.RandomBrightnessContrast(
            brightness_limit=(mag_ratio * 0.9, mag_ratio * 0.9),
            contrast_limit=0,
            p=1.0,
        ),
        # 3. Color (Saturation) adjustment
        A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(mag_ratio * 90),
            val_shift_limit=0,
            p=1.0,
        ),
        # 4. Sharpness
        A.Sharpen(
            alpha=(mag_ratio * 0.9, mag_ratio * 0.9),
            lightness=(1.0, 1.0),
            p=1.0,
        ),
        # 5. Rotate
        A.Rotate(
            limit=int(mag_ratio * 30),  # 0-30° at magnitude=10
            border_mode=0,
            p=1.0,
        ),
        # 6. ShearX
        A.Affine(
            shear={"x": (-mag_ratio * 30, mag_ratio * 30), "y": (0, 0)},
            p=1.0,
        ),
        # 7. ShearY
        A.Affine(
            shear={"x": (0, 0), "y": (-mag_ratio * 30, mag_ratio * 30)},
            p=1.0,
        ),
        # 8. TranslateX
        A.Affine(
            translate_percent={"x": (-mag_ratio * 0.45, mag_ratio * 0.45), "y": (0, 0)},
            p=1.0,
        ),
        # 9. TranslateY
        A.Affine(
            translate_percent={"x": (0, 0), "y": (-mag_ratio * 0.45, mag_ratio * 0.45)},
            p=1.0,
        ),
        # 10. AutoContrast (CLAHE approximation)
        A.CLAHE(
            clip_limit=(2.0, 2.0),
            tile_grid_size=(8, 8),
            p=1.0,
        ),
        # 11. Equalize
        A.Equalize(p=1.0),
        # 12. Invert
        A.InvertImg(p=1.0),
        # 13. Posterize
        A.Posterize(
            num_bits=max(1, 8 - int(mag_ratio * 4)),
            p=1.0,
        ),
        # 14. Solarize (Note: Albumentations Solarize uses different params)
        A.Solarize(p=1.0),
    ]

    return augmentation_pool


class RandAugment(A.BaseCompose):
    """
    RandAugment transform for Albumentations.

    Randomly selects N augmentation operations from a predefined pool
    and applies them sequentially with uniform magnitude M.

    Args:
        n: Number of augmentation operations to apply (default: 2)
        m: Magnitude of augmentations on scale 0-10 (default: 9)
           - 0: No augmentation (identity)
           - 5: Moderate augmentation
           - 10: Maximum augmentation

    Example:
        >>> rand_aug = RandAugment(n=2, m=9)
        >>> augmented = rand_aug(image=image_np)
        >>> image_augmented = augmented['image']
    """

    def __init__(self, n: int = 2, m: int = 9) -> None:
        """Initialize RandAugment with n ops and magnitude m."""
        self.n = n
        self.m = m
        self.augmentation_pool = get_randaugment_transforms(n_ops=n, magnitude=m)
        super().__init__([], p=1.0)  # Empty transforms list, we'll handle it manually

    def __call__(self, force_apply: bool = False, **kwargs) -> dict:
        """
        Apply N random augmentation operations.

        Args:
            force_apply: Not used (for compatibility)
            **kwargs: Must contain 'image' key with numpy array

        Returns:
            Dictionary with augmented 'image'
        """
        image = kwargs.get("image")
        if image is None:
            raise ValueError("RandAugment requires 'image' in kwargs")

        # Randomly select N operations
        selected_ops = np.random.choice(
            self.augmentation_pool, size=self.n, replace=False
        )

        # Apply selected operations sequentially
        for op in selected_ops:
            augmented = op(image=image)
            image = augmented["image"]

        return {"image": image}


def create_randaugment_pipeline(
    n: int = 2,
    m: int = 9,
    additional_transforms: List[A.BasicTransform] | None = None,
) -> A.Compose:
    """
    Create a complete augmentation pipeline with RandAugment.

    Args:
        n: Number of RandAugment operations (default: 2)
        m: Magnitude of augmentations (default: 9)
        additional_transforms: Optional list of transforms to apply after RandAugment
                               (e.g., normalization, tensor conversion)

    Returns:
        Albumentations Compose pipeline

    Example:
        >>> from albumentations.pytorch import ToTensorV2
        >>> pipeline = create_randaugment_pipeline(
        ...     n=2, m=9,
        ...     additional_transforms=[
        ...         A.Normalize(mean=[0.5071, 0.4867, 0.4408],
        ...                     std=[0.2675, 0.2565, 0.2761]),
        ...         ToTensorV2()
        ...     ]
        ... )
    """
    transforms = [RandAugment(n=n, m=m)]

    if additional_transforms:
        transforms.extend(additional_transforms)

    return A.Compose(transforms)
