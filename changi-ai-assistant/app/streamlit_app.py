import streamlit as st
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.inference import predict_cnn, predict_rnn, predict_resnet, CNN_CONFIG, RNN_CONFIG, RESNET_CONFIG

st.set_page_config(page_title="Changi Airport AI Assistant", page_icon="✈️", layout="wide")

# Combine all image models
ALL_IMAGE_MODELS = dict(
    **{f"CNN: {k}": ("cnn", k) for k in CNN_CONFIG.keys()},
    **{f"ResNet: {k}": ("resnet", k) for k in RESNET_CONFIG.keys()}
)

st.title("✈️ Changi Airport AI Assistant")
st.markdown("**Multi-Model Prototype:** CNN + ResNet + RNN")

tab1, tab2 = st.tabs(["🖼️ Image Classification (CNN)", "💬 Text Classification (RNN)"])

with tab1:
    st.subheader("Aircraft Image Classification")
    
    model_name = st.selectbox("Choose Model", list(ALL_IMAGE_MODELS.keys()))
    model_type, model_key = ALL_IMAGE_MODELS[model_name]
    
    uploaded = st.file_uploader("Upload an aircraft image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            image_path = tmp.name

        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    if model_type == "cnn":
                        preds = predict_cnn(image_path, model_key, top_k=5)
                    else:
                        preds = predict_resnet(image_path, model_key, top_k=5)
                    
                    best = preds[0]
                    st.success(f"**Top Prediction:** {best['label']}")
                    st.metric("Confidence", f"{best['confidence']:.1%}")

                    st.write("**Top-5 Predictions:**")
                    for i, p in enumerate(preds, 1):
                        st.write(f"{i}. {p['label']}: {p['confidence']:.1%}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure model files are in models/ directory")

with tab2:
    st.subheader("RNN: Text Classification")
    rnn_task = st.selectbox("Choose RNN task", list(RNN_CONFIG.keys()))
    text = st.text_area("Enter a passenger/staff message", height=120)

    if st.button("Run RNN Prediction"):
        if text.strip() == "":
            st.warning("Please type a message first.")
        else:
            try:
                out = predict_rnn(text, rnn_task)
                st.write(f"**Prediction:** {out['label']}")
                st.write(f"**Confidence:** {out['confidence']:.3f}")
            except FileNotFoundError as e:
                st.error(f"❌ {str(e)}")
                st.info("📝 Please follow the RNN_MODEL_SETUP.md instructions to export the models from your training notebook.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
