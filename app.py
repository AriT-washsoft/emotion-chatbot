
import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image

st.set_page_config(page_title="Emotion Detection Chatbot", layout="centered")

detector = FER(mtcnn=True)
camera_placeholder = st.empty()
frame_placeholder = st.empty()
emotion_placeholder = st.empty()
response_placeholder = st.empty()

def analyze_emotion(frame):
    emotion, score = None, 0
    try:
        result = detector.top_emotion(frame)
        if result:
            emotion, score = result
    except Exception as e:
        st.error(f"Error during emotion detection: {e}")
    return emotion, score

def generate_response(emotion):
    responses = {
        "happy": "You look happy! 😊 What's going on?",
        "sad": "I'm here for you. Want to talk about it?",
        "angry": "Take a deep breath. I'm listening.",
        "surprise": "Whoa! Something unexpected?",
        "fear": "You seem worried. How can I help?",
        "disgust": "Yikes, that didn’t look pleasant.",
        "neutral": "All calm, huh? Let’s chat!"
    }
    return responses.get(emotion, "I'm not sure how you feel. Tell me more!")

# Webcam stream
cap = cv2.VideoCapture(0)
frame_to_analyze = None

if not cap.isOpened():
    st.error("Unable to access webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", caption="Live Webcam Feed")
        key = cv2.waitKey(1)
        frame_to_analyze = frame_rgb
        break  # Stop after one frame to let Streamlit update

if st.button("Capture Emotion"):
    if frame_to_analyze is not None:
        emotion, score = analyze_emotion(frame_to_analyze)
        if emotion:
            emotion_placeholder.success(f"Emotion Detected: {emotion} ({score*100:.1f}%)")
            response_placeholder.info(generate_response(emotion))
        else:
            emotion_placeholder.warning("Couldn't detect any clear emotion.")
        frame_placeholder.image(frame_to_analyze, caption="Analyzed Frame", channels="RGB")
    else:
        st.warning("No frame to analyze.")

cap.release()
cv2.destroyAllWindows()
