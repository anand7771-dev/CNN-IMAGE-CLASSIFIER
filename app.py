"""
Streamlit Web App for Fashion-MNIST CNN Image Classifier
=========================================================
Interactive web interface to upload images and get real-time
predictions from the trained CNN model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from PIL import Image
import os
import urllib.request

# ─── Configuration ───────────────────────────────────────────────
MODEL_PATH = "fashion_mnist_cnn_model_v2.keras"
MODEL_URL = "https://huggingface.co/ananddev7771/CNN-IMAGE-CLASSIFIER/resolve/main/fashion_mnist_cnn_model_v2.keras"

CLASS_NAMES = [
    'T-shirt/Top 👕', 'Trouser 👖', 'Pullover 🧥', 'Dress 👗', 'Coat 🧥',
    'Sandal 👡', 'Shirt 👔', 'Sneaker 👟', 'Bag 👜', 'Ankle Boot 👢'
]
CLASS_NAMES_PLAIN = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]

st.set_page_config(
    page_title="Fashion-MNIST CNN Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1.1rem; }
    .prediction-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px; padding: 1.5rem;
        text-align: center; color: white; margin: 1rem 0;
    }
    .prediction-box h2 { color: white; font-size: 2rem; margin: 0; }
    .confidence-text { font-size: 1.2rem; opacity: 0.9; }
    .info-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.2rem; border-left: 4px solid #667eea; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model..."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return None
    try:
        return load_model(MODEL_PATH, compile=False)  # ← add compile=False
    except Exception as e:
        # Try legacy keras loading
        try:
            import keras
            return keras.saving.load_model(MODEL_PATH, compile=False)
        except Exception as e2:
            st.error(f"❌ Failed to load model: {e2}")
            return None

def preprocess_image(uploaded_file):
    """
    Preprocess uploaded image to match Fashion-MNIST format.
    Fashion-MNIST: 28x28 grayscale, light clothing on dark/black background.
    """
    from PIL import ImageOps

    img = Image.open(uploaded_file).convert('L')  # Grayscale

    # Center crop to square
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img_cropped = img.crop((left, top, left + min_dim, top + min_dim))

    # Add a small border padding (like Fashion-MNIST has)
    img_cropped = ImageOps.pad(img_cropped, (32, 32), color=0, centering=(0.5, 0.5))

    # Resize to 28x28
    img_resized = img_cropped.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_resized).astype('float32')

    # Smart inversion: check the CORNER pixels to detect background color
    # Fashion-MNIST backgrounds are black (dark), so if corners are light, invert
    corners = [
        img_array[0, 0], img_array[0, -1],
        img_array[-1, 0], img_array[-1, -1],
        img_array[0, 14], img_array[-1, 14],  # top/bottom center edges
    ]
    avg_corner = np.mean(corners)
    if avg_corner > 120:  # Background is light -> invert
        img_array = 255.0 - img_array

    # Contrast stretch to full range
    p_low, p_high = np.percentile(img_array, 2), np.percentile(img_array, 98)
    if p_high > p_low:
        img_array = np.clip((img_array - p_low) / (p_high - p_low) * 255.0, 0, 255)

    # Normalize to [0,1]
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Display version
    display_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8), mode='L')

    return img, img_array, display_img


def create_prediction_chart(predictions):
    colors = ['#667eea' if p < max(predictions) else '#11998e' for p in predictions]
    fig = go.Figure(go.Bar(
        x=predictions * 100, y=CLASS_NAMES, orientation='h',
        marker_color=colors,
        text=[f'{p:.1f}%' for p in predictions * 100],
        textposition='outside', textfont=dict(size=12)
    ))
    fig.update_layout(
        title=dict(text='Prediction Confidence (%)', font=dict(size=16)),
        xaxis=dict(title='Confidence (%)', range=[0, 105]),
        yaxis=dict(autorange='reversed'), height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def main():
    st.markdown("""
    <div class="main-header">
        <h1>👗 Fashion-MNIST CNN Classifier</h1>
        <p>Upload a clothing image and let the CNN identify it!</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📖 About")
        st.markdown("This app uses a **custom CNN** trained on **Fashion-MNIST** to classify clothing images.")
        st.markdown("### 🏷️ Supported Classes")
        for name in CLASS_NAMES:
            st.markdown(f"- {name}")
        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        st.markdown("- 3 Conv Blocks + BatchNorm\n- Data Augmentation\n- Global Average Pooling\n- Dropout Regularization")

    model = load_trained_model()
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check your MODEL_URL or re-upload the model.")
        st.info("💡 Make sure you have replaced `<your-username>` and `<repo-name>` in MODEL_URL inside app.py.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])
        if uploaded_file is not None:
            original_img, img_array, processed_img = preprocess_image(uploaded_file)
            st.image(original_img, caption="Uploaded Image", use_container_width=True)
            st.image(processed_img, caption="What the model sees (28×28)", width=150)

    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 Results")
            with st.spinner("Analyzing..."):
                predictions = model.predict(img_array, verbose=0)[0]

            idx = np.argmax(predictions)
            confidence = predictions[idx] * 100

            st.markdown(f"""
            <div class="prediction-box">
                <h2>{CLASS_NAMES[idx]}</h2>
                <p class="confidence-text">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            top3 = np.argsort(predictions)[::-1][:3]
            cols = st.columns(3)
            for i, j in enumerate(top3):
                with cols[i]:
                    st.metric(f"#{i+1}", CLASS_NAMES_PLAIN[j], f"{predictions[j]*100:.1f}%")

            fig = create_prediction_chart(predictions)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("### 🎯 Ready to Classify")
            st.markdown('<div class="info-card"><strong>Upload a clothing image</strong> to get started!</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
