import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings
import av
import cv2
import numpy as np
from fer import FER
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Emotion Detection Chatbot", layout="centered")

# Initialize FER detector
detector = FER()

# Title and Description
st.title("😊 Emotion Detection Chatbot")
st.write("This app uses your webcam to detect your emotion and responds with a chatbot-like message.")

# Store state variables
if "last_frame" not in st.session_state:
    st.session_state["last_frame"] = None

if "captured_frame" not in st.session_state:
    st.session_state["captured_frame"] = None

if "detected_emotion" not in st.session_state:
    st.session_state["detected_emotion"] = None


class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None
        self.last_emotion = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        return img


# Start the webcam stream
ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    video_transformer_factory=EmotionTransformer,
    async_processing=True,
)

# Button to capture emotion
if st.button("🎯 Capture Emotion"):
    if ctx.video_transformer and ctx.video_transformer.last_frame is not None:
        frame = ctx.video_transformer.last_frame
        st.session_state["captured_frame"] = frame

        # Save temp image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            cv2.imwrite(tmpfile.name, frame)
            try:
                result = detector.detect_emotions(frame)
                if result and "emotions" in result[0]:
                    emotions = result[0]["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    score = emotions[dominant_emotion]
                    st.session_state["detected_emotion"] = f"{dominant_emotion} ({score:.2f})"
                else:
                    st.session_state["detected_emotion"] = "Could not detect emotion"
            except Exception as e:
                st.session_state["detected_emotion"] = f"Emotion detection error: {e}"
    else:
        st.warning("No webcam frame detected yet.")

# Show detected emotion
if st.session_state["detected_emotion"]:
    st.markdown(f"**Detected Emotion:** `{st.session_state['detected_emotion']}`")

# Show captured frame
if st.session_state["captured_frame"] is not None:
    st.image(cv2.cvtColor(st.session_state["captured_frame"], cv2.COLOR_BGR2RGB), caption="Analyzed Frame", channels="RGB")