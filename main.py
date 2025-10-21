#!/usr/bin/env python3
# cifar_pipeline.py - Complete pipeline for CIFAR-10/100 data preparation, augmentation, training and evaluation

import argparse
import logging
import os
import random
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets

from scripts.data_augmentation import augment_dataset

# Import our custom modules
from scripts.data_download import (
    download_and_extract_cifar10_data,
    download_and_extract_cifar100_data,
)
from scripts.evaluation_metrics import (
    evaluate_model,
)
from scripts.model_architectures import create_model
from scripts.train_utils import (
    ModelEMA,
    define_loss_and_optimizer,
    load_data,
    load_transforms,
    save_checkpoint,
    save_metrics,
    train_epoch,
    validate_epoch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("cifar_pipeline.log")],
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        CUBLAS = "CUBLAS_WORKSPACE_CONFIG"
        if CUBLAS not in os.environ:
            os.environ[CUBLAS] = ":4096:8"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Training Pipeline")

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar100",
        help="Dataset to use (cifar10 or cifar100)",
    )

    # Data paths
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Base directory for data storage"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    # Data augmentation
    parser.add_argument(
        "--use_online_aug",
        action="store_true",
        default=True,
        help="Use online augmentation (dynamic, different every epoch, default: enabled). "
        "If disabled, use offline augmentation (pre-generated, larger dataset).",
    )
    parser.add_argument(
        "--no_online_aug",
        dest="use_online_aug",
        action="store_false",
        help="Disable online augmentation and use offline augmentation instead",
    )
    parser.add_argument(
        "--aug_count",
        type=int,
        default=3,
        help="Number of augmentations per image (for offline augmentation)",
    )
    parser.add_argument(
        "--aug_strength",
        type=str,
        choices=["light", "medium", "strong", "randaugment"],
        default="randaugment",
        help="Augmentation strategy. Options: "
        "light/medium/strong (traditional augmentation), "
        "randaugment (automatic, N=2 M=9, F1=0.8131, recommended)",
    )
    # use_randaugment removed - now controlled by aug_strength="randaugment"
    # This simplifies the interface and avoids confusion
    parser.add_argument(
        "--randaugment_n",
        type=int,
        default=2,
        help="RandAugment N: number of augmentation operations to apply (default: 2, range: 1-3)",
    )
    parser.add_argument(
        "--randaugment_m",
        type=int,
        default=9,
        help="RandAugment M: magnitude of augmentations (default: 9, range: 0-10). "
        "Higher values = stronger augmentation. Recommended: 9 for CIFAR-100.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.25,
        help="Mixup alpha parameter (default: 0.4). "
        "Recommended: 1.0 for CIFAR-100. "
        "Mixup mixes training examples to improve generalization. "
        "0.0=disabled, 0.2-0.4=light, 1.0=uniform (recommended). "
        "Note: Ignored if --use_cutmix is enabled.",
    )
    parser.add_argument(
        "--use_cutmix",
        action="store_true",
        default=True,
        help="Use CutMix instead of Mixup (recommended for CIFAR-100 to preserve local features). "
        "CutMix cuts and pastes patches between images, preserving local clarity. "
        "This is especially beneficial for 32x32 images where Mixup blurs fine-grained features.",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=0.65,
        help="CutMix alpha parameter (default: 1.0 = recommended for CIFAR-100). "
        "Controls the size distribution of cut regions. "
        "1.0 is the standard setting from the paper. "
        "Only used if --use_cutmix is enabled.",
    )

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
        ],
        default="wide_resnet28_10",
        help="Model architecture (all from scratch). "
        "Phase 1: wide_resnet28_10 (36.5M, F1=0.8131). "
        "Phase 2.7: pyramidnet110_270 (26M, paper: 83%%, target F1≥0.85). "
        "Others: resnet34/50, wide_resnet*, pyramidnet164, convnext, selfdistill.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for regularization. Recommended: 0.3 for Wide ResNet, 0.5 for ResNet",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="Stochastic Depth (DropPath) rate for Wide ResNet (0.0=disabled, 0.1=best for WRN-28-10). "
        "Randomly drops residual branches during training to reduce overfitting. "
        "Achieved F1=0.8131 with drop_path=0.1. Only effective for Wide ResNet models.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=True,
        help="Use torch.compile() for faster training (PyTorch 2.0+). "
        "Expected speedup: 5-15%% (first epoch will be slower due to compilation). "
        "Disable with --no_compile if encountering issues.",
    )
    parser.add_argument(
        "--no_compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile() optimization",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay (L2 penalty)"
    )

    # Optimizer and scheduler
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Optimizer: adam, adamw (recommended), or sgd",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["plateau", "cosine", "step", "onecycle"],
        default="cosine",
        help="LR scheduler: plateau, cosine (recommended), step, or onecycle",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0-1.0). 0.15 is recommended for CIFAR-100 with CE loss",
    )

    # Loss function
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["ce", "focal"],
        default="ce",
        help="Loss function: 'ce' (CrossEntropy with optional label smoothing) or 'focal' (Focal Loss for hard examples)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (default: 2.0). Higher values focus more on hard examples. Recommended: 1.0-3.0",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=False,
        help="Use class weights to focus on hard classes (seal, lizard, otter, etc.)",
    )

    # Mixed precision training
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (AMP) training (default: enabled)",
    )
    parser.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false",
        help="Disable AMP training",
    )

    # Gradient clipping
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping. 0 = no clipping. Recommended: 1.0",
    )

    # Self-Distillation (BYOT)
    parser.add_argument(
        "--use_self_distillation",
        action="store_true",
        default=False,
        help="Use self-distillation (Be Your Own Teacher). "
        "Only works with wide_resnet28_10_selfdistill model. "
        "Expected improvement: +3-4%% F1 (paper: arXiv:1905.08094).",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=3.0,
        help="Temperature for self-distillation (default: 3.0, lowered for stability). "
        "Range: 2-4. Higher = softer targets.",
    )
    parser.add_argument(
        "--distill_alpha",
        type=float,
        default=0.7,
        help="Weight for soft labels in self-distillation (default: 0.7, lowered for stability). "
        "loss = alpha * KL + (1-alpha) * CE. Range: 0.5-0.9.",
    )

    # EMA (Exponential Moving Average)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use Exponential Moving Average for model weights (default: enabled)",
    )
    parser.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (default: 0.9999). Higher = slower update",
    )

    # Learning rate warmup
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs for cosine scheduler (default: 5)",
    )

    # Checkpointing
    parser.add_argument(
        "--save_freq", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,  # 35 or 30 does not matter (in most cases)
        help="Early stopping patience. Increased to 20 to allow more training before stopping",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4, recommended for Windows)",
    )

    # Random seeds
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Ensemble training
    parser.add_argument(
        "--ensemble_seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for ensemble training (e.g., '42,43,44'). "
        "If specified, trains multiple models sequentially and ensembles them at evaluation. "
        "Example: --ensemble_seeds 42,43,44 trains 3 models and ensembles their predictions.",
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        default="results/ensemble",
        help="Directory to save ensemble models (default: results/ensemble)",
    )

    return parser.parse_args()


def collect_data(args):
    """Collect data"""
    logger.info(f"Collecting {args.dataset} dataset...")

    # Create the directory for our raw data if it doesn't already exist
    print("Preparing data directory...")
    raw_dir = args.data_dir + "/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Check and warn if mixed dataset exists
    train_dir = raw_dir + "/train"
    if os.path.exists(train_dir):
        num_classes = len(
            [
                d
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        )
        expected_classes = 10 if args.dataset == "cifar10" else 100
        if num_classes != expected_classes and num_classes > 0:
            print(
                f"⚠️  Warning: Found {num_classes} classes, but {args.dataset.upper()} should have {expected_classes}!"
            )
            print(
                f"   This may indicate mixed CIFAR-10/100 data. Consider cleaning: rm -rf {train_dir}"
            )
            raise ValueError(
                f"Data contamination detected: {num_classes} classes found, expected {expected_classes}. "
                f"Please delete '{train_dir}' and '{raw_dir.replace('/raw', '/augmented')}' to start fresh."
            )

    print("Setup complete.")

    if args.dataset == "cifar10":
        train_dataset, test_dataset = download_and_extract_cifar10_data(
            root_dir=raw_dir,
        )
    else:
        train_dataset, test_dataset = download_and_extract_cifar100_data(
            root_dir=raw_dir,
        )


def augment_data(args):
    """Prepare and augment data"""
    logger.info(f"Augmenting {args.dataset} dataset...")

    raw_data_dir = args.data_dir + "/raw/train/"
    augmented_data_dir = args.data_dir + "/augmented/train/"
    augmentations_per_image = args.aug_count
    augmentation_strength = args.aug_strength

    # --- Path Validation ---
    # Check if the raw data directory exists before proceeding.
    if not os.path.exists(raw_data_dir):
        print(f"❌ Error: Raw data directory '{raw_data_dir}' not found.")
        print("Please ensure you have run 'collect_data' first.")
    else:
        print(f"✅ Found raw data at: {raw_data_dir}")
        print(f"   Augmented data will be saved to: {augmented_data_dir}")
        print(f"   Number of augmentations per image: {augmentations_per_image}")
        print(f"   Augmentation strength: {augmentation_strength}")

    # Ensure the raw data directory exists before running
    if os.path.exists(raw_data_dir):
        print("🚀 Starting data augmentation...")
        augment_dataset(
            input_dir=raw_data_dir,
            output_dir=augmented_data_dir,
            augmentations_per_image=augmentations_per_image,
            augmentation_strength=augmentation_strength,
            use_cutmix=args.use_cutmix,
        )
        print("\n🎉 Data augmentation completed successfully!")
    else:
        print("Skipping augmentation process due to missing raw data directory.")

    return augmented_data_dir


def build_model(args) -> nn.Module:
    """Build the model (from scratch)"""
    if args.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100

    logger.info(
        f"Creating {args.model} model with {num_classes} classes, "
        f"dropout={args.dropout}, device={args.device} (training from scratch)..."
    )

    model = create_model(
        num_classes=num_classes,
        device=args.device,
        model_type=args.model,
        dropout_rate=args.dropout,
        drop_path_rate=args.drop_path_rate,
    )

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable"
    )

    # Apply torch.compile() for performance optimization (PyTorch 2.0+)
    if hasattr(torch, "compile") and args.use_compile:
        logger.info("🚀 Compiling model with torch.compile() for faster training...")
        logger.info("   (First epoch will be slower due to compilation overhead)")
        try:
            # Use fullgraph=False (default) for better compatibility
            # fullgraph=True may fail with dynamic control flow (e.g., dropout)
            # Strategy: Try inductor first (best performance), fallback to aot_eager if Triton unavailable
            import platform

            if platform.system() == "Windows":
                # Windows doesn't support Triton, use AOT backend directly
                logger.info(
                    "   Detected Windows: using backend='aot_eager' (Triton not available)"
                )
                compiled_model = torch.compile(
                    model,
                    backend="aot_eager",  # Windows-compatible backend with AOT optimization
                    fullgraph=False,
                )
            else:
                # Linux: Try inductor (needs Triton), fallback to aot_eager
                try:
                    logger.info(
                        "   Detected Linux: trying backend='inductor' with mode='reduce-overhead'"
                    )
                    compiled_model = torch.compile(
                        model,
                        backend="inductor",  # Explicitly use inductor
                        mode="reduce-overhead",
                        fullgraph=False,  # Allow graph breaks for robustness
                    )
                except Exception as e:
                    # Triton not available on Linux, fallback to aot_eager
                    logger.warning(
                        f"   Inductor backend failed (likely missing Triton): {e}"
                    )
                    logger.info("   Falling back to backend='aot_eager'")
                    compiled_model = torch.compile(
                        model,
                        backend="aot_eager",
                        fullgraph=False,
                    )
            # torch.compile() returns a wrapper that's still callable as nn.Module
            model = cast(nn.Module, compiled_model)
            logger.info("✅ Model compiled successfully!")
            logger.info(
                "   Expected speedup: 3-8%% (aot_eager), 5-15%% (inductor with Triton)"
            )
        except Exception as e:
            logger.warning(f"⚠️  torch.compile() failed: {e}")
            logger.warning("   Falling back to eager mode (no compilation)")

    return model


def train(args, model: nn.Module):
    # Disable `label-smoothing` while using `CutMix` or `Mixup`
    if (args.use_cutmix and args.cutmix_alpha > 0) or args.mixup_alpha > 0:
        args.label_smoothing = 0.0
        if args.use_cutmix and args.cutmix_alpha > 0 and args.mixup_alpha > 0:
            logger.info("--label_smoothing disabled while using Mixup + CutMix")
            logger.info("Using ADAPTIVE augmentation strategy (category-aware):")
            logger.info(
                "  • Detail-sensitive (human/small animals): Mixup only (alpha=0.4)"
            )
            logger.info("  • Local-feature (mechanical/plants): 80% CutMix, 20% Mixup")
            logger.info("  • Mixed-strategy (others): 30% Mixup, 70% CutMix")
        elif args.use_cutmix and args.cutmix_alpha > 0:
            logger.info("--label_smoothing disabled while using CutMix")
        elif args.mixup_alpha > 0:
            logger.info("--label_smoothing disabled while using Mixup")

    # Determine data directory based on augmentation strategy
    if args.use_online_aug:
        data_dir = args.data_dir + "/raw/train"
        logger.info("Using ONLINE augmentation (dynamic, different every epoch)")
    else:
        data_dir = args.data_dir + "/augmented/train"
        logger.info(
            f"Using OFFLINE augmentation (pre-generated, {args.aug_count}x augmentations per image)"
        )

    # Load data first to get steps_per_epoch for OneCycleLR
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
    )
    steps_per_epoch = len(train_loader)

    # Get number of classes
    num_classes = 10 if args.dataset == "cifar10" else 100

    # Define loss, optimizer, and scheduler with new options
    criterion, optimizer, scheduler = define_loss_and_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        label_smoothing=args.label_smoothing,
        num_epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        use_class_weights=args.use_class_weights,
        num_classes=num_classes,
    )

    # Initialize EMA if enabled
    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=args.device)
        logger.info(f"EMA enabled with decay={args.ema_decay}")

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # save best F1
    patience_counter = 0

    # Lists to store training history for later plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create directories for saving models and results if they don't exist
    os.makedirs(args.output_dir + "/models", exist_ok=True)
    os.makedirs(args.output_dir + "/results", exist_ok=True)

    print(
        f"Training configured for {args.num_epochs} epochs with early stopping patience of {args.early_stopping_patience}."
    )
    print(f"Optimizer: {args.optimizer}, Scheduler: {args.scheduler}")
    print(
        f"Label smoothing: {args.label_smoothing}, Mixed precision: {args.use_amp}, "
        f"Gradient clipping: {args.max_grad_norm if args.max_grad_norm > 0 else 'disabled'}"
    )
    print(
        f"EMA: {args.use_ema}, Warmup epochs: {args.warmup_epochs if args.scheduler == 'cosine' else 'N/A (OneCycle has built-in warmup)'}"
    )
    # Log data augmentation strategy
    if args.use_cutmix and args.cutmix_alpha > 0:
        print(f"CutMix: Enabled (alpha={args.cutmix_alpha})")
        if args.mixup_alpha > 0:
            print(f"Mixup: ALSO Enabled (alpha={args.mixup_alpha})")
    elif args.mixup_alpha > 0:
        print(f"Mixup: Enabled (alpha={args.mixup_alpha})")
    else:
        print("Mixup/CutMix: Disabled")

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Train for one epoch with new options
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            use_amp=args.use_amp,
            max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            ema=ema,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            use_cutmix=args.use_cutmix,
            use_self_distill=args.use_self_distillation,
            distill_temperature=args.distill_temperature,
            distill_alpha=args.distill_alpha,
        )

        # Validate the model (use EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow()
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, args.device
        )
        if ema is not None:
            ema.restore()

        # Update learning rate based on scheduler type
        if args.scheduler == "plateau":
            # Type narrowing: ReduceLROnPlateau requires a metric
            cast(torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler).step(val_loss)
        elif args.scheduler in ["cosine", "step"]:
            scheduler.step()
        # OneCycleLR updates per batch in train_epoch

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.num_epochs}:")
        print(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}"
        )
        print(
            f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}"
        )

        # Check for improvement and save the best model
        # Two-tier strategy: Prioritize Loss, then F1
        # 1. If Loss doesn't increase, save (regardless of F1)
        # 2. If Loss increases but F1 improves, also save

        save_model = False
        save_reason = ""
        if val_loss <= best_val_loss:
            # Priority 1: Loss didn't increase - save model
            old_best_val_loss = best_val_loss
            best_val_loss = val_loss
            best_val_f1 = val_f1
            save_model = True
            save_reason = (
                f"Loss improved/stable: {val_loss:.4f} ≤ {old_best_val_loss:.4f}"
            )
        elif val_f1 > best_val_f1:
            # Priority 2: Loss increased but F1 improved - still save
            old_best_val_f1 = best_val_f1
            best_val_f1 = val_f1
            best_val_loss = val_loss
            save_model = True
            save_reason = (
                f"Loss increased but F1 improved: {val_f1:.4f} > {old_best_val_f1:.4f}"
            )

        if save_model:
            patience_counter = 0

            # Prepare checkpoint (with EMA weights if enabled)
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            # Add EMA state if enabled
            if ema is not None:
                checkpoint["ema_shadow"] = ema.shadow

            save_checkpoint(
                checkpoint,
                args.output_dir + "/models/best_model.pth",
            )
            print(f"  ↳ Validation improved ({save_reason}). Saving best model!")
        else:
            patience_counter += 1
            print(
                f"  ↳ No improvement (Loss: {val_loss:.4f} > {best_val_loss:.4f}, F1: {val_f1:.4f} ≤ {best_val_f1:.4f}). "
                f"Early stopping counter: {patience_counter}/{args.early_stopping_patience}"
            )

        # Check for early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    print("\nTraining completed!")

    # Load the best model checkpoint saved during training
    checkpoint = torch.load(args.output_dir + "/models/best_model.pth")
    model.load_state_dict(checkpoint["state_dict"])

    # Restore EMA weights if they were saved
    if ema is not None and "ema_shadow" in checkpoint:
        ema.shadow = checkpoint["ema_shadow"]
        ema.apply_shadow()
        print("Loaded EMA weights from checkpoint")

    # Retrieve details from the checkpoint
    best_epoch = checkpoint["epoch"]
    best_val_loss_loaded = checkpoint["best_val_loss"]
    best_val_f1_loaded = checkpoint.get(
        "best_val_f1", 0.0
    )  # get best F1 (compatible with old checkpoint)

    print(
        f"Loaded best model from epoch {best_epoch} "
        f"with validation F1-score {best_val_f1_loaded:.4f} (loss {best_val_loss_loaded:.4f})"
    )

    # Save the final model's state_dict for easy use in evaluation/inference
    torch.save(model.state_dict(), args.output_dir + "/models/final_model.pth")
    print(
        f"Final model state_dict saved to '{args.output_dir}/models/final_model.pth'."
    )

    return model, best_val_loss


def evaluate(args, model: nn.Module):
    """Evaluate the model on test data"""
    # Load the test dataset from the specified directory
    test_data_dir = args.data_dir + "/raw/test"
    test_dataset = datasets.ImageFolder(
        root=test_data_dir, transform=load_transforms(dataset_type=args.dataset)
    )

    # Validate test dataset class count
    expected_classes = 10 if args.dataset == "cifar10" else 100
    actual_classes = len(test_dataset.classes)
    if actual_classes != expected_classes:
        raise ValueError(
            f"⚠️ Test data contamination detected!\n"
            f"   Expected {expected_classes} classes for {args.dataset.upper()}, "
            f"but found {actual_classes} classes.\n"
            f"   Please delete '{test_data_dir}' and re-run data collection to fix.\n"
            f"   Command: python main.py --dataset {args.dataset}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Set the model to evaluation mode
    model.eval()

    # Get number of classes
    num_classes = 10 if args.dataset == "cifar10" else 100

    # Define loss function (same as training for consistency)
    criterion, _, _ = define_loss_and_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        use_class_weights=args.use_class_weights,
        num_classes=num_classes,
    )

    # Evaluate the model
    test_loss, test_accuracy, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, criterion, args.device
    )
    metrics_str = classification_report(
        all_labels, all_preds, target_names=test_dataset.classes
    )

    save_metrics(metrics=metrics_str)


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Check if ensemble training is requested
    if args.ensemble_seeds is not None:
        # Ensemble mode: train multiple models with different seeds
        ensemble_main(args)
    else:
        # Standard mode: single model training
        standard_main(args)


def standard_main(args):
    """Standard single-model training pipeline"""
    # Set random seeds
    set_random_seeds(args.seed)

    # Print configuration
    logger.info("Starting CIFAR pipeline with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Collect data
    collect_data(args)

    # Data augmentation (based on strategy)
    if args.use_online_aug:
        logger.info(
            "Using ONLINE augmentation strategy (dynamic, different every epoch)."
        )
    else:
        logger.info(
            f"Using OFFLINE augmentation strategy (pre-generating {args.aug_count}x augmented data)."
        )
        augment_data(args)

    # Build model
    model = build_model(args)
    # Train
    train(args, model)
    # Evaluate
    evaluate(args, model)


def ensemble_main(args):
    """Ensemble training pipeline - trains multiple models and creates ensemble"""
    import os

    from scripts.model_architectures import EnsembleModel

    # Parse ensemble seeds
    seeds = [int(s.strip()) for s in args.ensemble_seeds.split(",")]

    logger.info("=" * 80)
    logger.info(f"ENSEMBLE MODE: Training {len(seeds)} models with seeds {seeds}")
    logger.info(f"Models will be saved to: {args.ensemble_dir}")
    logger.info("=" * 80)

    # Collect data once
    collect_data(args)

    # Train each model and collect them
    trained_model_list = []
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training model {i + 1}/{len(seeds)} with seed={seed}")
        logger.info(f"{'=' * 80}\n")

        # Update seed
        args.seed = seed
        set_random_seeds(seed)

        # Create model-specific output directory
        model_output_dir = os.path.join(args.ensemble_dir, f"model_seed{seed}")
        os.makedirs(model_output_dir, exist_ok=True)

        # Temporarily change output_dir
        original_output_dir = args.output_dir
        args.output_dir = model_output_dir

        # Build and train model
        model = build_model(args)
        train(args, model)

        # Keep the trained model (already loaded with best weights from train())
        trained_model_list.append(model)

        # Restore original output_dir
        args.output_dir = original_output_dir

    # Create ensemble model combining all trained models
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CREATING ENSEMBLE MODEL: Combining {len(trained_model_list)} models")
    logger.info(f"{'=' * 80}\n")

    ensemble_model = EnsembleModel(trained_model_list)

    # Log ensemble parameters
    total_params = sum(p.numel() for p in ensemble_model.parameters())
    logger.info(f"Ensemble model parameters: {total_params / 1e6:.2f}M total")
    logger.info(
        f"  (= {len(trained_model_list)} models × {total_params / len(trained_model_list) / 1e6:.2f}M each)"
    )

    # Use original evaluate() function on the ensemble model
    logger.info("\nEvaluating ensemble model using standard pipeline...")
    evaluate(args, ensemble_model)


if __name__ == "__main__":
    main()
