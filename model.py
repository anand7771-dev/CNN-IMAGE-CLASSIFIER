"""
CNN Model Architecture for Fashion-MNIST Image Classification
==============================================================
Custom Convolutional Neural Network with BatchNormalization,
Dropout regularization, and Global Average Pooling.
"""

from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a custom CNN model for image classification.

    Architecture:
        - 3 Convolutional Blocks (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout)
        - Global Average Pooling
        - Dense classifier head

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name="FashionMNIST_CNN")

    # ==================== Block 1 ====================
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv1a'))
    model.add(layers.BatchNormalization(name='bn1a'))
    model.add(layers.Activation('relu', name='relu1a'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', name='conv1b'))
    model.add(layers.BatchNormalization(name='bn1b'))
    model.add(layers.Activation('relu', name='relu1b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='drop1'))

    # ==================== Block 2 ====================
    model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2a'))
    model.add(layers.BatchNormalization(name='bn2a'))
    model.add(layers.Activation('relu', name='relu2a'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2b'))
    model.add(layers.BatchNormalization(name='bn2b'))
    model.add(layers.Activation('relu', name='relu2b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(layers.Dropout(0.30, name='drop2'))

    # ==================== Block 3 ====================
    model.add(layers.Conv2D(128, (3, 3), padding='same', name='conv3a'))
    model.add(layers.BatchNormalization(name='bn3a'))
    model.add(layers.Activation('relu', name='relu3a'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', name='conv3b'))
    model.add(layers.BatchNormalization(name='bn3b'))
    model.add(layers.Activation('relu', name='relu3b'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(layers.Dropout(0.40, name='drop3'))

    # ==================== Classifier Head ====================
    model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    model.add(layers.Dense(256, name='fc1'))
    model.add(layers.BatchNormalization(name='bn_fc1'))
    model.add(layers.Activation('relu', name='relu_fc1'))
    model.add(layers.Dropout(0.5, name='drop_fc'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    return model


def get_compiled_model(learning_rate=0.001):
    """
    Build and compile the CNN model with Adam optimizer.

    Args:
        learning_rate: Initial learning rate for Adam optimizer

    Returns:
        Compiled Keras model ready for training
    """
    from tensorflow.keras.optimizers import Adam

    model = build_cnn_model()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    model = get_compiled_model()
    model.summary()
