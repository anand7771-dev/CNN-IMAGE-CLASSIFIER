<div align="center">

# 👗 Fashion Image Classifier

### A Deep Learning CNN trained on Fashion-MNIST to classify clothing items with high accuracy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[🚀 Get Started](#-getting-started) · [🧠 Model Architecture](#️-model-architecture) · [📊 Results](#-results) · [🔮 Roadmap](#-roadmap)

</div>

---

## 📌 Overview

This project implements an end-to-end **Convolutional Neural Network (CNN)** pipeline to classify grayscale fashion images into **10 distinct clothing categories**. Trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, the model uses modern deep learning techniques including batch normalization, data augmentation, and dropout regularization to achieve robust performance.

> **Fashion-MNIST** is a dataset of Zalando's article images — 60,000 training examples and 10,000 test examples, each a 28×28 grayscale image associated with one of 10 classes.

---

## 🏷️ Supported Classes

| Label | Class | Description |
|:-----:|-------|-------------|
| `0` | 👕 T-shirt / Top | Casual upper wear |
| `1` | 👖 Trouser | Pants & trousers |
| `2` | 🧥 Pullover | Sweaters & pullovers |
| `3` | 👗 Dress | Full-length dresses |
| `4` | 🧥 Coat | Outerwear & coats |
| `5` | 👡 Sandal | Open footwear |
| `6` | 👔 Shirt | Formal shirts |
| `7` | 👟 Sneaker | Athletic shoes |
| `8` | 👜 Bag | Handbags & purses |
| `9` | 👢 Ankle Boot | Ankle boots |

---

## 🧠 Model Architecture

The CNN is designed to extract rich spatial features while staying efficient and generalizable:

```
Input (28×28×1)
     │
     ▼
┌─────────────────────────────┐
│  Conv Block 1               │
│  Conv2D → BN → ReLU → Pool  │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Conv Block 2               │
│  Conv2D → BN → ReLU → Pool  │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Conv Block 3               │
│  Conv2D → BN → ReLU → Pool  │
└─────────────────────────────┘
     │
     ▼
  Global Average Pooling
     │
     ▼
  Dropout (Regularization)
     │
     ▼
  Dense → Softmax (10 classes)
```

| Component | Detail |
|-----------|--------|
| **Convolutional Blocks** | 3 blocks with Batch Normalization |
| **Pooling** | Global Average Pooling (no Flatten) |
| **Regularization** | Dropout to prevent overfitting |
| **Data Augmentation** | Random flips, rotations, shifts |
| **Activation** | ReLU (hidden), Softmax (output) |

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | Fashion-MNIST (via `tensorflow.keras.datasets`) |

---

## 📂 Project Structure

```
cnn-image-classifier/
│
├── 📄 train.py               # Model training script
├── 📄 model.py               # CNN architecture definition
├── 📄 predict.py             # Run predictions on new images
├── 📄 utils.py               # Helper & utility functions
├── 📄 app.py                 # (Optional) Streamlit web interface
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md              # Project documentation
│
└── 📁 results/
    ├── 🖼️  confusion_matrix.png     # Per-class performance breakdown
    ├── 📈  training_history.png     # Accuracy & loss curves
    └── 🖼️  sample_predictions.png   # Visual prediction examples
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anand7771-dev/CNN-IMAGE-CLASSIFIER.git
cd CNN-IMAGE-CLASSIFIER
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train.py
```

> The Fashion-MNIST dataset will be downloaded automatically on first run.

### 4️⃣ Run Predictions

```bash
python predict.py
```

---

## 📊 Results

The model demonstrates strong performance across all 10 fashion categories:

| Metric | Value |
|--------|-------|
| ✅ Training Accuracy | High |
| ✅ Validation Accuracy | High |
| ✅ Generalization | Strong (low overfitting) |

**Output Visualizations:**

- 📊 **Confusion Matrix** — per-class accuracy breakdown
- 📈 **Training History** — loss & accuracy curves over epochs
- 🖼️ **Sample Predictions** — visual comparison of true vs. predicted labels

---

## 💡 Key Highlights

- ✔️ **End-to-end pipeline** — from raw data to evaluation
- ✔️ **Modular codebase** — clean separation of training, model, and prediction logic
- ✔️ **Regularization techniques** — batch norm + dropout for better generalization
- ✔️ **Data augmentation** — improves model robustness on unseen images
- ✔️ **Beginner-friendly** — well-commented code and clear project structure
- ✔️ **Extensible** — easy to swap datasets or upgrade the architecture

---

## 🔮 Roadmap

- [ ] 🚀 Implement Transfer Learning (ResNet, EfficientNet)
- [ ] 🌐 Deploy as a Streamlit Web App
- [ ] ⚡ Hyperparameter tuning with Keras Tuner
- [ ] 📱 Real-time image classification via webcam
- [ ] 🧪 Add unit tests for model components
- [ ] 📦 Export model to TFLite for mobile deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

<div align="center">

**Anand Dev**

[![GitHub](https://img.shields.io/badge/GitHub-anand7771--dev-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anand7771-dev)

</div>

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*It motivates continued development and helps others discover the project.*

</div>
