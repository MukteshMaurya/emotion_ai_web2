import threading

import av
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from twilio.rest import Client
from transformers import ViTForImageClassification, ViTImageProcessor

st.set_page_config(page_title="Live Facial Expression Recognition", page_icon="😊", layout="wide")

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"


@st.cache_data(ttl=3000, show_spinner=False)
def get_ice_servers(account_sid, auth_token):
    client = Client(account_sid, auth_token)
    token = client.tokens.create(ttl=3600)
    return token.ice_servers


def create_rtc_configuration():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    except KeyError:
        st.error(
            "Twilio credentials are missing. Add TWILIO_ACCOUNT_SID and "
            "TWILIO_AUTH_TOKEN in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    try:
        return {"iceServers": get_ice_servers(account_sid, auth_token)}
    except Exception as e:
        st.error("Could not create the WebRTC ICE configuration.")
        st.exception(e)
        st.stop()


rtc_configuration = create_rtc_configuration()


@st.cache_resource
def load_emotion_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV Haar Cascade could not be loaded.")
    return detector


with st.spinner("Loading AI model..."):
    processor, model, device = load_emotion_model()
    face_cascade = load_face_detector()

model_lock = threading.Lock()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        inputs = processor(images=pil_face, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with model_lock:
            with torch.inference_mode():
                logits = model(**inputs).logits

        probabilities = F.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probabilities).item()
        label = model.config.id2label[pred_idx]
        confidence = probabilities[pred_idx].item()

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        display_text = f"{label.upper()}: {confidence * 100:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_top = max(0, y - text_height - 12)

        cv2.rectangle(
            image,
            (x, text_top),
            (x + text_width + 10, y),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            display_text,
            (x + 5, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return av.VideoFrame.from_ndarray(image, format="bgr24")


st.title("😊 Live Facial Expression Recognition")
st.write(
    "Real-time facial expression detection using OpenCV face detection "
    "and a Vision Transformer (ViT)."
)
st.info(
    "Click START, allow camera permission in your browser, and the "
    "application will process the live video."
)

webrtc_ctx = webrtc_streamer(
    key="live-facial-expression",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 10}},
        "audio": False,
    },
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("🟢 Camera connected — live emotion detection is running.")
else:
    st.warning("🔴 Camera is stopped. Click START above.")

with st.sidebar:
    st.header("About")
    st.write("Live facial expression recognition from the browser webcam.")
    st.write("**Face detection:** OpenCV Haar Cascade")
    st.write("**Emotion recognition:** ViT")
    st.write("**Live video:** WebRTC")
    st.write("**Network traversal:** Twilio TURN/STUN")
    st.divider()
    st.write("**Model**")
    st.code(MODEL_NAME)
    st.write("**Device**")
    st.code(str(device))
    st.divider()
    st.caption(
        "cv2.VideoCapture(0) is intentionally not used. The browser sends "
        "the webcam stream through WebRTC."
    )
