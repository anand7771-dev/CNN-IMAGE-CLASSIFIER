"""
Prediction Script for Fashion-MNIST CNN Classifier
====================================================
Load a trained model and predict the class of an input image.
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]

MODEL_PATH = "fashion_mnist_cnn_model.keras"


def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict(image_path, model_path=MODEL_PATH):
    """Predict the class of an image using the trained CNN model."""
    model = load_model(model_path)
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    return CLASS_NAMES[predicted_class], confidence, predictions


def main():
    parser = argparse.ArgumentParser(description='Predict image class using trained Fashion-MNIST CNN')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Fashion-MNIST CNN Image Classifier")
    print("=" * 50)

    predicted_class, confidence, predictions = predict(args.image, args.model)

    print(f"\n  Image      : {args.image}")
    print(f"  Prediction : {predicted_class}")
    print(f"  Confidence : {confidence * 100:.2f}%")

    top_k_indices = np.argsort(predictions)[::-1][:args.top_k]
    print(f"\n  Top-{args.top_k} Predictions:")
    print("  " + "-" * 35)
    for rank, idx in enumerate(top_k_indices, 1):
        bar = "█" * int(predictions[idx] * 30)
        print(f"  {rank}. {CLASS_NAMES[idx]:12s} {predictions[idx] * 100:6.2f}% {bar}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
