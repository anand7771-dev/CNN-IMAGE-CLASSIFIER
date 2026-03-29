import keras
import numpy as np

# Load with current version
model = keras.models.load_model("fashion_mnist_cnn_model.keras")

# Re-save in compatible format
model.save("fashion_mnist_cnn_model_v2.keras")
print("Done! Model re-saved successfully.")