"""
Training Script for Fashion-MNIST CNN Classifier
==================================================
Downloads Fashion-MNIST (~11 MB), applies data augmentation,
trains the CNN model, and saves results.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from model import get_compiled_model
from utils import plot_training_history, plot_confusion_matrix, plot_sample_predictions

# ─── Configuration ───────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 20
MODEL_SAVE_PATH = "fashion_mnist_cnn_model.keras"
RESULTS_DIR = "results"

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]


def load_and_preprocess_data():
    """Load Fashion-MNIST dataset and preprocess it."""
    print("=" * 60)
    print("  Loading Fashion-MNIST Dataset...")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Reshape to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    print(f"  Training samples  : {x_train.shape[0]}")
    print(f"  Test samples      : {x_test.shape[0]}")
    print(f"  Image shape       : {x_train.shape[1:]}")
    print(f"  Number of classes : 10")
    print("=" * 60)

    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat


def create_data_augmentation():
    """Create data augmentation generator for training."""
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    return datagen


def get_callbacks():
    """Create training callbacks."""
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def train():
    """Main training function."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = load_and_preprocess_data()

    # Build model
    print("\n  Building CNN Model...")
    model = get_compiled_model()
    model.summary()

    # Data augmentation
    datagen = create_data_augmentation()
    datagen.fit(x_train)

    # Train
    print("\n" + "=" * 60)
    print("  Starting Training...")
    print("=" * 60 + "\n")

    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test_cat),
        callbacks=get_callbacks(),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("  Evaluating Model on Test Set...")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\n  Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")

    # Generate predictions for plots
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = y_test.flatten()

    # Save plots
    print("\n  Generating training plots...")
    plot_training_history(history, save_path=os.path.join(RESULTS_DIR, "training_history.png"))
    plot_confusion_matrix(y_true_classes, y_pred_classes, CLASS_NAMES,
                          save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_sample_predictions(x_test, y_true_classes, y_pred_classes, CLASS_NAMES,
                            save_path=os.path.join(RESULTS_DIR, "sample_predictions.png"))

    # Save final model
    model.save(MODEL_SAVE_PATH)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Model saved to       : {MODEL_SAVE_PATH}")
    print(f"  Results saved to     : {RESULTS_DIR}/")
    print(f"  Final Test Accuracy  : {test_accuracy * 100:.2f}%")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train()
