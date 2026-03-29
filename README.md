# 🧠 CNN-Based Image Classification System

## CIFAR-10 Image Classifier using Convolutional Neural Networks

A complete Computer Vision project that builds, trains, and deploys a custom CNN for classifying images into 10 categories using the CIFAR-10 dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## 📖 Overview

This project implements a **custom Convolutional Neural Network (CNN)** from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset. The project includes:

- A well-designed CNN architecture with **BatchNormalization** and **Dropout**
- **Data augmentation** for improved generalization
- Training with **learning rate scheduling** and **early stopping**
- Comprehensive **evaluation metrics** (accuracy, confusion matrix, classification report)
- An interactive **Streamlit web application** for real-time image classification

---

## 📊 Dataset

**CIFAR-10** consists of 60,000 color images (32×32 pixels) across 10 classes:

| Class | Examples |
|-------|----------|
| ✈️ Airplane | Aircraft, jets |
| 🚗 Automobile | Cars, sedans |
| 🐦 Bird | Various bird species |
| 🐱 Cat | Domestic cats |
| 🦌 Deer | Wild deer |
| 🐶 Dog | Various dog breeds |
| 🐸 Frog | Frogs, toads |
| 🐴 Horse | Horses |
| 🚢 Ship | Ships, boats |
| 🚚 Truck | Trucks, lorries |

- **Training set:** 50,000 images
- **Test set:** 10,000 images
- **Image size:** 32 × 32 × 3 (RGB)

---

## 🏗️ Model Architecture

```
Input (32×32×3)
    │
    ├── Conv Block 1: 2×Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ├── Conv Block 2: 2×Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.30)
    │
    ├── Conv Block 3: 2×Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.40)
    │
    ├── Global Average Pooling
    │
    ├── Dense(256) → BatchNorm → ReLU → Dropout(0.50)
    │
    └── Dense(10, softmax) → Output
```

**Key Design Choices:**
- **BatchNormalization** after every convolution for stable training
- **Progressive Dropout** (0.25 → 0.30 → 0.40 → 0.50) to prevent overfitting
- **Global Average Pooling** instead of Flatten to reduce parameters
- **Data Augmentation** (rotation, shift, flip, zoom) for better generalization

---

## 📁 Project Structure

```
cnn-image-classifier/
├── model.py           # CNN model architecture definition
├── train.py           # Training script with data augmentation
├── predict.py         # CLI prediction script
├── app.py             # Streamlit web application
├── utils.py           # Visualization utilities
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── results/           # Generated after training
    ├── training_history.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# 1. Navigate to project directory
cd cnn-image-classifier

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Download the CIFAR-10 dataset automatically
- Train the CNN for 25 epochs with data augmentation
- Save the trained model as `cifar10_cnn_model.keras`
- Generate training plots in the `results/` directory

### 2. Make Predictions (CLI)

```bash
python predict.py --image path/to/your/image.jpg
```

### 3. Launch Web App

```bash
streamlit run app.py
```

This opens an interactive web interface where you can upload images and get real-time predictions.

### 4. View Model Architecture

```bash
python model.py
```

---

## 📈 Results

After training, the following outputs are generated in the `results/` directory:

| Output | Description |
|--------|-------------|
| `training_history.png` | Training/validation accuracy and loss curves |
| `confusion_matrix.png` | 10×10 confusion matrix heatmap |
| `sample_predictions.png` | Grid of sample predictions with true vs predicted labels |

**Expected Performance:** ~85-90% test accuracy (varies with training run)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **TensorFlow/Keras** | Deep learning framework for CNN |
| **NumPy** | Numerical computing |
| **Matplotlib** | Training visualization |
| **Seaborn** | Confusion matrix heatmap |
| **Scikit-learn** | Classification metrics |
| **Streamlit** | Interactive web application |
| **Plotly** | Interactive charts in web app |
| **Pillow** | Image loading and preprocessing |

---

## 🔑 Key Concepts Demonstrated

1. **Convolutional Neural Networks (CNNs)** — Feature extraction from images
2. **Data Augmentation** — Artificially expanding training data
3. **Batch Normalization** — Stabilizing and accelerating training
4. **Dropout Regularization** — Preventing overfitting
5. **Transfer of learned features** — Hierarchical feature learning
6. **Learning Rate Scheduling** — Adaptive learning rate
7. **Early Stopping** — Preventing overfitting via validation monitoring

---

## 📝 License

This project is created for educational purposes as part of a Computer Vision course.
