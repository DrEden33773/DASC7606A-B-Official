import logging
from pathlib import Path
from typing import Optional, Tuple

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CIFAR10Downloader:
    """
    A configurable class to download and prepare the CIFAR-10 dataset.

    Supports custom root directories, transforms, and verbose reporting.
    Also supports saving images to organized train/test/classname folders.
    """

    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(
        self,
        root_dir: str = "data/raw",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        log_stats: bool = True,
    ):
        """
        Initialize the CIFAR-10 downloader.

        Args:
            root_dir: Base directory to store downloaded dataset and extracted images.
            transform: Optional torchvision transform pipeline. If None, uses default.
            download: Whether to download if not present.
            log_stats: Whether to log dataset statistics after loading.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self._get_default_transform()
        self.download = download
        self.log_stats = log_stats

        # Ensure base directory exists
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_default_transform() -> transforms.Compose:
        """Return default transform: ToTensor + Normalize to [-1, 1]."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_datasets(self) -> Tuple[CIFAR10, CIFAR10]:
        """
        Load CIFAR-10 train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            RuntimeError: If dataset fails to download or load.
        """
        try:
            logger.info("Loading CIFAR-10 training dataset...")
            train_dataset = CIFAR10(
                root=str(self.root_dir),
                train=True,
                download=self.download,
                transform=self.transform,
            )

            logger.info("Loading CIFAR-10 test dataset...")
            test_dataset = CIFAR10(
                root=str(self.root_dir),
                train=False,
                download=self.download,
                transform=self.transform,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR-10 dataset: {e}") from e

        if self.log_stats:
            self._log_dataset_stats(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _log_dataset_stats(self, train_dataset: CIFAR10, test_dataset: CIFAR10) -> None:
        """Log formatted dataset statistics."""
        total = len(train_dataset) + len(test_dataset)
        separator = "=" * 50

        logger.info("\n" + separator)
        logger.info("✅ CIFAR-10 Dataset Loaded Successfully!")
        logger.info(separator)
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   • Training samples: {len(train_dataset):,}")
        logger.info(f"   • Test samples: {len(test_dataset):,}")
        logger.info(f"   • Total samples: {total:,}")
        logger.info("   • Image size: 32x32 pixels")
        logger.info("   • Color channels: 3 (RGB)")
        logger.info(f"   • Number of classes: {len(self.CLASS_NAMES)}")
        logger.info(f"   • Classes: {', '.join(self.CLASS_NAMES)}")
        logger.info(separator)

    def save_images_to_folders(
        self, train_dataset: CIFAR10, test_dataset: CIFAR10
    ) -> None:
        """
        Save CIFAR-10 images as PNG files into organized folders:
        - raw/train/<class_name>/
        - raw/test/<class_name>/

        Args:
            train_dataset: The loaded CIFAR-10 training dataset.
            test_dataset: The loaded CIFAR-10 test dataset.
        """
        # Define output directories
        train_image_dir = self.root_dir / "train"
        test_image_dir = self.root_dir / "test"

        for class_name in self.CLASS_NAMES:
            (train_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_image_dir / class_name).mkdir(parents=True, exist_ok=True)

        logger.info("Saving training images...")
        self._save_dataset_images(train_dataset, train_image_dir)

        logger.info("Saving test images...")
        self._save_dataset_images(test_dataset, test_image_dir)

        logger.info("✅ All images saved successfully!")

    def _save_dataset_images(self, dataset: CIFAR10, base_dir: Path) -> None:
        """
        Helper to save images from a dataset into class-named subfolders.

        Args:
            dataset: The CIFAR-10 dataset (train or test).
            base_dir: Root directory (train or test) to save images under.
        """
        for idx, (image, label) in enumerate(dataset):
            # If transform includes ToTensor, convert back to PIL for saving
            if isinstance(image, transforms.functional.Tensor):
                # Denormalize if normalized
                if image.min() < 0:  # Assume normalized to [-1, 1]
                    image = image * 0.5 + 0.5
                image = transforms.functional.to_pil_image(image)

            class_name = self.CLASS_NAMES[label]
            file_path = base_dir / class_name / f"{idx}.png"
            image.save(file_path)


class CIFAR100Downloader:
    """
    A configurable class to download and prepare the CIFAR-100 dataset.

    Supports custom root directories, transforms, and verbose reporting.
    Also supports saving images to organized train/test/classname folders.
    """

    # CIFAR-100 has 100 fine-grained classes
    CLASS_NAMES = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    def __init__(
        self,
        root_dir: str = "data/raw",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        log_stats: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._get_default_transform()
        self.download = download
        self.log_stats = log_stats

        # Ensure base directory exists
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_default_transform() -> transforms.Compose:
        """Return default transform: ToTensor + Normalize to [-1, 1]."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_datasets(self) -> Tuple[CIFAR100, CIFAR100]:
        """
        Load CIFAR-100 train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            RuntimeError: If dataset fails to download or load.
        """
        try:
            logger.info("Loading CIFAR-100 training dataset...")
            train_dataset = CIFAR100(
                root=str(self.root_dir),
                train=True,
                download=self.download,
                transform=self.transform,
            )

            logger.info("Loading CIFAR-100 test dataset...")
            test_dataset = CIFAR100(
                root=str(self.root_dir),
                train=False,
                download=self.download,
                transform=self.transform,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR-100 dataset: {e}") from e

        if self.log_stats:
            self._log_dataset_stats(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _log_dataset_stats(
        self, train_dataset: CIFAR100, test_dataset: CIFAR100
    ) -> None:
        """Log formatted dataset statistics."""
        total = len(train_dataset) + len(test_dataset)
        separator = "=" * 50

        logger.info("\n" + separator)
        logger.info("✅ CIFAR-100 Dataset Loaded Successfully!")
        logger.info(separator)
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   • Training samples: {len(train_dataset):,}")
        logger.info(f"   • Test samples: {len(test_dataset):,}")
        logger.info(f"   • Total samples: {total:,}")
        logger.info("   • Image size: 32x32 pixels")
        logger.info("   • Color channels: 3 (RGB)")
        logger.info(f"   • Number of classes: {len(self.CLASS_NAMES)}")
        logger.info("   • Classes: Too many to list (100 classes)")
        logger.info(separator)

    def save_images_to_folders(
        self, train_dataset: CIFAR100, test_dataset: CIFAR100
    ) -> None:
        """
        Save CIFAR-100 images as PNG files into organized folders:
        - raw/train/<class_name>/
        - raw/test/<class_name>/

        Args:
            train_dataset: The loaded CIFAR-100 training dataset.
            test_dataset: The loaded CIFAR-100 test dataset.
        """
        # Define output directories
        train_image_dir = self.root_dir / "train"
        test_image_dir = self.root_dir / "test"

        for class_name in self.CLASS_NAMES:
            (train_image_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_image_dir / class_name).mkdir(parents=True, exist_ok=True)

        logger.info("Saving training images...")
        self._save_dataset_images(train_dataset, train_image_dir)

        logger.info("Saving test images...")
        self._save_dataset_images(test_dataset, test_image_dir)

        logger.info("✅ All images saved successfully!")

    def _save_dataset_images(self, dataset: CIFAR100, base_dir: Path) -> None:
        """
        Helper to save images from a dataset into class-named subfolders.

        Args:
            dataset: The CIFAR-100 dataset (train or test).
            base_dir: Root directory (train or test) to save images under.
        """
        for idx, (image, label) in enumerate(dataset):
            # If transform includes ToTensor, convert back to PIL for saving
            if isinstance(image, transforms.functional.Tensor):
                # Denormalize if normalized
                if image.min() < 0:  # Assume normalized to [-1, 1]
                    image = image * 0.5 + 0.5
                image = transforms.functional.to_pil_image(image)

            class_name = self.CLASS_NAMES[label]
            file_path = base_dir / class_name / f"{idx}.png"
            image.save(file_path)


def download_and_extract_cifar10_data(
    root_dir: str = "data/raw",
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
    log_stats: bool = True,
    save_images: bool = True,
) -> Tuple[CIFAR10, CIFAR10]:
    """
    Convenience function to download and extract CIFAR-10 dataset.

    Optionally saves images to organized train/test folders.

    Args:
        root_dir: Directory to store dataset.
        transform: Optional transform pipeline.
        download: Whether to download if not present.
        log_stats: Whether to print dataset statistics.
        save_images: Whether to save images as PNG files in train/test/class folders.

    Returns:
        Tuple of (train_dataset, test_dataset)

    Example:
        >>> train_data, test_data = download_and_extract_cifar10_data(save_images=True)
        >>> print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    """
    downloader = CIFAR10Downloader(
        root_dir=root_dir,
        transform=transform,
        download=download,
        log_stats=log_stats,
    )
    train_dataset, test_dataset = downloader.load_datasets()

    if save_images:
        downloader.save_images_to_folders(train_dataset, test_dataset)

    return train_dataset, test_dataset


def download_and_extract_cifar100_data(
    root_dir: str = "data/raw",
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
    log_stats: bool = True,
    save_images: bool = True,
) -> Tuple[CIFAR100, CIFAR100]:
    """
    Convenience function to download and extract CIFAR-100 dataset.

    Optionally saves images to organized train/test folders.

    Args:
        root_dir: Directory to store dataset.
        transform: Optional transform pipeline.
        download: Whether to download if not present.
        log_stats: Whether to print dataset statistics.
        save_images: Whether to save images as PNG files in train/test/class folders.

    Returns:
        Tuple of (train_dataset, test_dataset)

    Example:
        >>> train_data, test_data = download_and_extract_cifar100_data(save_images=True)
        >>> print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    """
    downloader = CIFAR100Downloader(
        root_dir=root_dir,
        transform=transform,
        download=download,
        log_stats=log_stats,
    )
    train_dataset, test_dataset = downloader.load_datasets()

    if save_images:
        downloader.save_images_to_folders(train_dataset, test_dataset)

    return train_dataset, test_dataset
