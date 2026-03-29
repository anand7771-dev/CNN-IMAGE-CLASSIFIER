"""
Streamlit Web App for Fashion-MNIST CNN Image Classifier (Demo Version)
=======================================================================
This version removes TensorFlow dependency and uses demo predictions
for successful deployment on Streamlit Cloud.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os

# ─── Configuration ───────────────────────────────────────────────
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

# ─── UI Styling ───────────────────────────────────────────────
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
.prediction-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 12px; padding: 1.5rem;
    text-align: center; color: white; margin: 1rem 0;
}
.info-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 1.2rem; border-left: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# ─── Image Preprocessing ───────────────────────────────────────────────
def preprocess_image(uploaded_file):
    from PIL import ImageOps

    img = Image.open(uploaded_file).convert('L')

    w, h = img.size
    min_dim = min(w, h)
    img = img.crop((0, 0, min_dim, min_dim))

    img = ImageOps.pad(img, (32, 32), color=0)
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    display_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))

    return img, img_array, display_img

# ─── Chart ───────────────────────────────────────────────
def create_prediction_chart(predictions):
    fig = go.Figure(go.Bar(
        x=predictions * 100,
        y=CLASS_NAMES,
        orientation='h'
    ))
    fig.update_layout(height=400)
    return fig

# ─── Main App ───────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>👗 Fashion-MNIST CNN Classifier</h1>
        <p>Upload a clothing image and get predictions!</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📖 About")
        st.markdown("Demo version (no TensorFlow)")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Image")

        if uploaded_file:
            original_img, img_array, processed_img = preprocess_image(uploaded_file)
            st.image(original_img, caption="Uploaded Image")
            st.image(processed_img, caption="Processed Image")

    with col2:
        if uploaded_file:
            with st.spinner("Analyzing..."):
                predictions = np.random.rand(10)
                predictions = predictions / np.sum(predictions)

            idx = np.argmax(predictions)
            confidence = predictions[idx] * 100

            st.markdown(f"""
            <div class="prediction-box">
                <h2>{CLASS_NAMES[idx]}</h2>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            fig = create_prediction_chart(predictions)
            st.plotly_chart(fig)

        else:
            st.markdown("Upload an image to start")

if __name__ == "__main__":
    main()
```
