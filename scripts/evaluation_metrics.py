import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


def top_k_accuracy(true_labels, probabilities, k=5):
    """
    Calculate top-k accuracy
    Args:
        true_labels: True labels
        probabilities: Prediction probabilities for each class
        k: Top-k value (default 5)
    Returns:
        Top-k accuracy percentage
    """
    # Get top-k predictions for each sample
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]

    # Check if true label is in top-k predictions
    correct = 0
    for i, true_label in enumerate(true_labels):
        if true_label in top_k_preds[i]:
            correct += 1

    return (correct / len(true_labels)) * 100


def plot_precision_recall_curves(true_labels, probabilities, class_names):
    """
    Plot precision-recall curves for multiclass classification
    Args:
        true_labels: True labels
        probabilities: Prediction probabilities for each class
        class_names: List of class names
    """
    # Binarize the output
    true_labels_bin = label_binarize(true_labels, classes=range(len(class_names)))

    # Compute precision-recall curve for each class
    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(
            true_labels_bin[:, i], probabilities[:, i]
        )
        pr_auc[i] = auc(recall[i], precision[i])

    # Plot all precision-recall curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", len(class_names))

    for i, color in zip(range(len(class_names)), colors(range(len(class_names)))):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"PR curve of {class_names[i]} (area = {pr_auc[i]:.2f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)


def plot_calibration_curve(true_labels, probabilities, class_names, n_bins=10):
    """
    Plot calibration curve to assess model confidence vs accuracy
    Args:
        true_labels: True labels
        probabilities: Prediction probabilities for each class
        class_names: List of class names
        n_bins: Number of bins for calibration curve
    """
    plt.figure(figsize=(10, 6))

    # Get predicted probabilities and whether prediction is correct
    pred_probs = np.max(probabilities, axis=1)
    pred_correct = np.argmax(probabilities, axis=1) == true_labels

    # Bin the predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate accuracy and confidence for each bin
    accuracy_bins = []
    confidence_bins = []

    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])
        if np.sum(mask) > 0:
            accuracy_bins.append(np.mean(pred_correct[mask]))
            confidence_bins.append(np.mean(pred_probs[mask]))
        else:
            accuracy_bins.append(0)
            confidence_bins.append(bin_centers[i])

    accuracy_bins = np.array(accuracy_bins)
    confidence_bins = np.array(confidence_bins)

    # Plot
    plt.plot(confidence_bins, accuracy_bins, "o-", label="Model calibration")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
    Returns:
        loss: Average loss
        accuracy: Accuracy percentage
        all_preds: All predictions
        all_labels: All true labels
        all_probs: All prediction probabilities
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    loss = running_loss / total
    accuracy = 100.0 * correct / total

    return (
        loss,
        accuracy,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """
    Plot a confusion matrix
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)


def plot_roc_curves(true_labels, probabilities, class_names):
    """
    Plot ROC curves for multiclass classification
    Args:
        true_labels: True labels
        probabilities: Prediction probabilities for each class
        class_names: List of class names
    """
    # Binarize the output
    true_labels_bin = label_binarize(true_labels, classes=range(len(class_names)))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = plt.cm.get_cmap("tab10", len(class_names))

    for i, color in zip(range(len(class_names)), colors(range(len(class_names)))):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")


def visualize_predictions(model, dataloader, device, class_names, num_samples=10):
    """
    Visualize correct and incorrect predictions
    Args:
        model: The model
        dataloader: DataLoader
        device: Device
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    model.eval()
    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.size(0)):
                if predicted[i] == labels[i] and len(correct_samples) < num_samples:
                    correct_samples.append(
                        (inputs[i].cpu(), labels[i].cpu(), predicted[i].cpu())
                    )
                elif predicted[i] != labels[i] and len(incorrect_samples) < num_samples:
                    incorrect_samples.append(
                        (inputs[i].cpu(), labels[i].cpu(), predicted[i].cpu())
                    )

                if (
                    len(correct_samples) >= num_samples
                    and len(incorrect_samples) >= num_samples
                ):
                    break
            else:
                continue
            break

    # Plot correct predictions
    plt.figure(figsize=(15, 6))
    plt.suptitle("Correct Predictions")
    for i, (img, true_label, pred_label) in enumerate(correct_samples):
        plt.subplot(2, 5, i + 1)
        img = img * 0.5 + 0.5  # Unnormalize
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/correct_predictions.png")
    plt.show()

    # Plot incorrect predictions
    plt.figure(figsize=(15, 6))
    plt.suptitle("Incorrect Predictions")
    for i, (img, true_label, pred_label) in enumerate(incorrect_samples):
        plt.subplot(2, 5, i + 1)
        img = img * 0.5 + 0.5  # Unnormalize
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/incorrect_predictions.png")
    plt.show()
