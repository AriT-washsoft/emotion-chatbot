import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from fer import FER
import cv2

st.set_page_config(page_title="Emotion Detection Chatbot", layout="centered")
st.title("🎭 Emotion Detection Chatbot")
st.markdown("This app uses your webcam to detect your facial emotion and responds accordingly.")

EMOTION_RESPONSES = {
    "happy": "I'm glad to see you're happy! 😊",
    "sad": "I'm here for you. What's wrong? 😢",
    "angry": "Take a deep breath. I'm listening. 😠",
    "surprise": "Whoa! That surprised look says a lot! 😲",
    "fear": "It’s okay to be scared. Let’s talk it out. 😨",
    "disgust": "Something bothering you? Tell me more. 😖",
    "neutral": "Just chilling? Let's chat! 😐",
}

class EmotionDetectorTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = FER()
        self.last_frame = None
        self.last_emotion = "Not detected"
        self.last_score = 0.0
        self.display_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        result = self.detector.detect_emotions(img)
        if result:
            top_result = result[0]
            emotions = top_result["emotions"]
            if emotions:
                self.last_emotion = max(emotions, key=emotions.get)
                self.last_score = emotions[self.last_emotion]
        self.display_frame = img.copy()
        return img

ctx = webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetectorTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

if ctx.video_transformer:
    st.markdown(f"**Detected Emotion:** `{ctx.video_transformer.last_emotion}`")
    st.markdown(f"**Confidence:** `{ctx.video_transformer.last_score * 100:.2f}%`")

    if st.button("Capture Emotion"):
        st.image(ctx.video_transformer.display_frame, caption="Captured Frame", channels="BGR", use_column_width=True)
        emotion = ctx.video_transformer.last_emotion
        st.markdown(f"### 💬 Chatbot Response:")
        st.info(EMOTION_RESPONSES.get(emotion, "Hmm, I'm not sure how to respond to that."))