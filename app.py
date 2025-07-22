import streamlit as st
from deepface import DeepFace
import cv2
import tempfile
import os
import sys
print("Python version:", sys.version)

st.set_page_config(page_title="Emotion Detection Chatbot", layout="centered")
st.title("🧠 Emotion Detection Chatbot")

# Upload or capture image
st.subheader("1. Capture or Upload Image")
image_source = st.radio("Select Image Source", ["Upload Image", "Use Webcam"])

image_path = None
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

elif image_source == "Use Webcam":
    st.warning("Webcam capture not supported in Streamlit Cloud/Render. Use a local setup.")

# Detect emotion
emotion_result = None
if image_path and st.button("Analyze Emotion"):
    with st.spinner("Analyzing emotion..."):
        try:
            analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)
            emotion_result = analysis[0]["dominant_emotion"]
            st.success(f"Detected Emotion: **{emotion_result.upper()}**")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Chatbot section
if emotion_result:
    st.subheader("2. Chatbot Response")
    if emotion_result.lower() == "happy":
        st.write("😊 I'm glad you're feeling happy! Keep smiling!")
    elif emotion_result.lower() == "sad":
        st.write("😢 It's okay to feel sad. I'm here if you want to talk.")
    elif emotion_result.lower() == "angry":
        st.write("😠 Take a deep breath. Want to vent a little?")
    elif emotion_result.lower() == "surprise":
        st.write("😲 Wow! Something unexpected?")
    elif emotion_result.lower() == "fear":
        st.write("😨 Don't worry, you're safe here.")
    elif emotion_result.lower() == "disgust":
        st.write("🤢 Yikes! Want to tell me more?")
    else:
        st.write("😐 I'm not sure how you're feeling. Want to share?")
