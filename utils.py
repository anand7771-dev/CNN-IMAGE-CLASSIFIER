"""
Utility Functions for Visualization
=====================================
Provides plotting functions for training history, confusion matrix,
and sample predictions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#FF5722')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2196F3')
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#FF5722')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true: True labels (integer encoded)
        y_pred: Predicted labels (integer encoded)
        class_names: List of class name strings
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion matrix saved to: {save_path}")

    # Also print classification report
    print("\n  📋 Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_true, y_pred, target_names=class_names)
    for line in report.split('\n'):
        print(f"  {line}")


def plot_sample_predictions(x_test, y_true, y_pred, class_names, num_samples=16,
                            save_path="sample_predictions.png"):
    """
    Plot a grid of sample predictions with true vs predicted labels.

    Args:
        x_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    rows = int(np.ceil(num_samples / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(x_test[idx].squeeze(), cmap='gray')
        axes[i].axis('off')

        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        is_correct = y_true[idx] == y_pred[idx]
        color = '#4CAF50' if is_correct else '#F44336'
        symbol = '✓' if is_correct else '✗'

        axes[i].set_title(
            f"{symbol} {pred_label}\n(True: {true_label})",
            fontsize=9, color=color, fontweight='bold'
        )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Sample predictions saved to: {save_path}")
