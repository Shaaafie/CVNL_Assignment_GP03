import streamlit as st
import tempfile

from src.inference import predict_cnn, predict_rnn, CNN_CONFIG, RNN_CONFIG

st.set_page_config(page_title="Changi AI Prototype", layout="centered")
st.title("Changi Airport AI Prototype (2 CNN + 2 RNN)")

tab1, tab2 = st.tabs(["🖼️ Image Classification (CNN)", "💬 Text Classification (RNN)"])

with tab1:
    st.subheader("CNN: Aircraft Image Classification")
    cnn_task = st.selectbox("Choose CNN task", list(CNN_CONFIG.keys()))
    uploaded = st.file_uploader("Upload an aircraft image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            image_path = tmp.name

        if st.button("Run CNN Prediction"):
            preds = predict_cnn(image_path, cnn_task, top_k=3)
            best = preds[0]
            st.write(f"**Prediction:** {best['label']}")
            st.write(f"**Confidence:** {best['confidence']:.3f}")

            st.write("**Top-3:**")
            for p in preds:
                st.write(f"- {p['label']}: {p['confidence']:.3f}")

with tab2:
    st.subheader("RNN: Text Classification")
    rnn_task = st.selectbox("Choose RNN task", list(RNN_CONFIG.keys()))
    text = st.text_area("Enter a passenger/staff message", height=120)

    if st.button("Run RNN Prediction"):
        if text.strip() == "":
            st.warning("Please type a message first.")
        else:
            out = predict_rnn(text, rnn_task)
            st.write(f"**Prediction:** {out['label']}")
            st.write(f"**Confidence:** {out['confidence']:.3f}")
