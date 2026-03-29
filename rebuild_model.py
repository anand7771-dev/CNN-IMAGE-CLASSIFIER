"""
Rebuild model with exact architecture for keras 2.15 compatibility
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
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


# Step 1: Load old model weights only
print("Loading old model...")
old_model = tf.keras.models.load_model("fashion_mnist_cnn_model.keras", compile=False)

# Step 2: Build fresh compatible model
print("Building new compatible model...")
new_model = build_cnn_model()
new_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 3: Transfer weights
print("Transferring weights...")
new_model.set_weights(old_model.get_weights())

# Step 4: Save in H5 format (universally compatible)
new_model.save("fashion_mnist_compatible.h5")
print("✅ Done! fashion_mnist_compatible.h5 saved successfully.")
