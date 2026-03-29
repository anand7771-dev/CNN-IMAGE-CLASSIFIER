👕 Fashion Image Classifier using CNN
📌 About

This project is a deep learning-based image classifier that uses a custom Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset to classify clothing images.

The model can accurately recognize different types of fashion items from grayscale images.

🧠 Supported Classes

The model classifies images into the following categories:

👕 T-shirt/Top
👖 Trouser
🧥 Pullover
👗 Dress
🧥 Coat
👡 Sandal
👔 Shirt
👟 Sneaker
👜 Bag
👢 Ankle Boot
🏗️ Model Architecture

The CNN model is designed with the following components:

🔹 3 Convolutional Blocks + Batch Normalization
🔹 Data Augmentation for better generalization
🔹 Global Average Pooling
🔹 Dropout Regularization to prevent overfitting
⚙️ Tech Stack
Python 🐍
TensorFlow / Keras
NumPy
Matplotlib
🚀 Features

✔️ Custom CNN model for image classification
✔️ Training and evaluation pipeline
✔️ Confusion matrix visualization
✔️ Training history plots
✔️ Prediction on custom images

📂 Project Structure
cnn-image-classifier/
│── train.py
│── model.py
│── predict.py
│── utils.py
│── app.py
│── requirements.txt
│── README.md
│
└── results/
    ├── confusion_matrix.png
    ├── training_history.png
    └── sample_predictions.png
▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train the model
python train.py
3️⃣ Run predictions
python predict.py
📊 Results
📈 Model achieves good accuracy on Fashion-MNIST
📊 Confusion matrix available in /results
🖼️ Sample predictions visualized
💡 Future Improvements
Use Transfer Learning (ResNet / EfficientNet)
Deploy using Streamlit Web App
Improve accuracy with hyperparameter tuning
🙌 Author

Anand Dev
🔗 https://github.com/anand7771-dev

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!