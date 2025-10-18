import logging
import random
from pathlib import Path
from typing import List, Literal, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
        self,
        augmentations_per_image: int = 5,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        augmentation_strength: Literal["light", "medium", "strong"] = "medium",
        use_cutmix: bool = False,
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
            augmentation_strength: Strength of augmentation pipeline. Options:
                - "light": Basic augmentations (original, for CIFAR-10)
                - "medium": Moderate augmentations (default)
                - "strong": Heavy augmentations (recommended for CIFAR-100)
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions
        self.augmentation_strength = augmentation_strength

        self._set_seed()

        # Define Albumentations pipeline based on strength
        self.transform = self._get_transform_pipeline(augmentation_strength, use_cutmix)

    def _get_transform_pipeline(
        self,
        strength: Literal["light", "medium", "strong"],
        use_cutmix: bool = False,
    ) -> A.Compose:
        """
        Get augmentation pipeline based on specified strength.

        Args:
            strength: Augmentation strength level

        Returns:
            Albumentations Compose object with appropriate transforms

        Raises:
            ValueError: If strength is not recognized
        """
        if strength == "light":
            # Original pipeline - suitable for CIFAR-10
            return A.Compose(
                [
                    A.Rotate(limit=15, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=0,
                        p=0.8,
                        border_mode=0,  # cv2.BORDER_CONSTANT
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
                ]
            )

        elif strength == "medium":
            # Enhanced pipeline with more diversity
            return A.Compose(
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
            )

        elif strength == "strong":
            # Strong pipeline optimized for CIFAR-100
            return A.Compose(
                [
                    # Geometric transformations
                    A.Rotate(limit=25, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.2,
                        scale_limit=0.2,
                        rotate_limit=20,
                        p=0.8,
                        border_mode=0,
                    ),
                    # Advanced geometric augmentations
                    A.OneOf(
                        [
                            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                            A.GridDistortion(p=0.3),
                            A.OpticalDistortion(distort_limit=0.3, p=0.3),
                        ],
                        p=0.3,
                    ),
                    # Color augmentations (critical for distinguishing similar classes)
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
                                r_shift_limit=20,
                                g_shift_limit=20,
                                b_shift_limit=20,
                                p=0.5,
                            ),
                            A.ChannelShuffle(p=0.2),
                        ],
                        p=0.5,
                    ),
                    # Blur and noise
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                            A.MotionBlur(blur_limit=9, p=0.4),
                            A.MedianBlur(blur_limit=7, p=0.3),
                            A.GaussNoise(p=0.3),
                        ],
                        p=0.5,
                    ),
                    # Brightness and contrast
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.6
                    ),
                ]
                + (
                    [
                        # Cutout/CoarseDropout for regularization
                        A.CoarseDropout(
                            num_holes_range=(1, 2),
                            hole_height_range=(6, 12),
                            hole_width_range=(6, 12),
                            p=0.5,
                        ),
                    ]
                    if not use_cutmix
                    else []
                )
                + [
                    # Additional pixel-level augmentations
                    A.OneOf(
                        [
                            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
                            A.RandomToneCurve(scale=0.1, p=0.3),
                        ],
                        p=0.3,
                    ),
                ]
            )

        else:
            raise ValueError(
                f"Unknown augmentation_strength: {strength}. "
                f"Available options: 'light', 'medium', 'strong'"
            )

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image using Albumentations.

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Augment all images in input directory and save to output directory.

        Preserves folder structure. Skips files that fail to load.

        Args:
            input_dir: Path to input directory with class subfolders.
            output_dir: Path to output directory for augmented images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        # Create progress bar for image processing
        progress_bar = tqdm(
            image_files,
            desc="🖼️  Processing images",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for img_path in progress_bar:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            # Update progress bar with current file info
            class_name = img_path.parent.name
            file_name = img_path.name
            progress_bar.set_postfix(
                {
                    "Class": class_name,
                    "File": file_name[:15] + "..."
                    if len(file_name) > 15
                    else file_name,
                    "Augmented": count,
                }
            )

            # Determine output subdirectory
            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            # Save original if requested
            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            # Generate and save augmented versions
            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{img_path.name}"
                augmented.save(target_dir / aug_name)
                count += 1

                # Update the augmented count in real-time
                progress_bar.set_postfix(
                    {
                        "Class": class_name,
                        "File": file_name[:15] + "..."
                        if len(file_name) > 15
                        else file_name,
                        "Augmented": count,
                    }
                )

        progress_bar.close()
        logger.info(
            f"✅ Augmentation completed! Generated {count} augmented images from {len(image_files)} originals. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
    augmentation_strength: Literal["light", "medium", "strong"] = "medium",
    use_cutmix: bool = False,
) -> None:
    """
    Backward-compatible wrapper for legacy code with enhanced augmentation options.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
        augmentation_strength: Strength of augmentation pipeline. Options:
            - "light": Basic augmentations (suitable for CIFAR-10)
            - "medium": Moderate augmentations (default)
            - "strong": Heavy augmentations (recommended for CIFAR-100)
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image,
        seed=seed,
        save_original=True,
        augmentation_strength=augmentation_strength,
        use_cutmix=use_cutmix,
    )
    augmenter.process_directory(input_dir, output_dir)


# ============================================================================
# RandAugment - Automatic Augmentation Search (Phase 1.5)
# ============================================================================


def get_randaugment_transforms(
    n_ops: int = 2,
    magnitude: int = 9,
) -> List[A.BasicTransform]:
    """
    Get RandAugment transform operations for CIFAR.

    Args:
        n_ops: Number of operations to apply per image (default: 2)
        magnitude: Magnitude of augmentations on scale 0-10 (default: 9)

    Returns:
        List of Albumentations transforms
    """
    mag_ratio = magnitude / 10.0

    augmentation_pool: List[A.BasicTransform] = [
        A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(mag_ratio * 0.9, mag_ratio * 0.9), p=1.0
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(mag_ratio * 0.9, mag_ratio * 0.9), contrast_limit=0, p=1.0
        ),
        A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(mag_ratio * 90),
            val_shift_limit=0,
            p=1.0,
        ),
        A.Sharpen(
            alpha=(mag_ratio * 0.9, mag_ratio * 0.9), lightness=(1.0, 1.0), p=1.0
        ),
        A.Rotate(limit=int(mag_ratio * 30), border_mode=0, p=1.0),
        A.Affine(shear={"x": (-mag_ratio * 30, mag_ratio * 30), "y": (0, 0)}, p=1.0),
        A.Affine(shear={"x": (0, 0), "y": (-mag_ratio * 30, mag_ratio * 30)}, p=1.0),
        A.Affine(
            translate_percent={"x": (-mag_ratio * 0.45, mag_ratio * 0.45), "y": (0, 0)},
            p=1.0,
        ),
        A.Affine(
            translate_percent={"x": (0, 0), "y": (-mag_ratio * 0.45, mag_ratio * 0.45)},
            p=1.0,
        ),
        A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0),
        A.Equalize(p=1.0),
        A.InvertImg(p=1.0),
        A.Posterize(num_bits=max(1, 8 - int(mag_ratio * 4)), p=1.0),
        A.Solarize(p=1.0),
    ]

    return augmentation_pool


class RandAugment(A.BaseCompose):
    """
    RandAugment transform for Albumentations.

    Randomly selects N augmentation operations from a predefined pool.
    Achieved F1=0.8131 on CIFAR-100 when used as primary augmentation.
    """

    def __init__(self, n: int = 2, m: int = 9) -> None:
        self.n = n
        self.m = m
        self.augmentation_pool = get_randaugment_transforms(n_ops=n, magnitude=m)
        super().__init__([], p=1.0)

    def __call__(self, force_apply: bool = False, **kwargs) -> dict:
        image = kwargs.get("image")
        if image is None:
            raise ValueError("RandAugment requires 'image' in kwargs")

        # Randomly select N operations using random.sample for type compatibility
        selected_ops = random.sample(self.augmentation_pool, k=self.n)

        for op in selected_ops:
            augmented = op(image=image)
            image = augmented["image"]

        return {"image": image}
