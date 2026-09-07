import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from transformers import ViTForImageClassification, ViTImageProcessor


st.set_page_config(
    page_title="Live Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
)

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


@st.cache_resource
def load_emotion_model():
    print("Loading ViT emotion model...")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Model loaded on:", device)

    return processor, model, device


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade.")

    print("Face detector loaded successfully.")

    return face_cascade


processor, model, device = load_emotion_model()
face_cascade = load_face_detector()

model_lock = threading.Lock()


class EmotionVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_detections = []
        self.frame_skip = 6

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if self.frame_count % self.frame_skip == 0:

            new_detections = []

            for (x, y, w, h) in faces:

                face_roi = image[y:y + h, x:x + w]

                if face_roi.size == 0:
                    continue

                face_rgb = cv2.cvtColor(
                    face_roi,
                    cv2.COLOR_BGR2RGB
                )

                pil_face = Image.fromarray(face_rgb)

                inputs = processor(
                    images=pil_face,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with model_lock:
                    with torch.no_grad():
                        logits = model(**inputs).logits

                probabilities = F.softmax(
                    logits,
                    dim=-1
                )[0]

                pred_idx = torch.argmax(
                    probabilities
                ).item()

                label = model.config.id2label[pred_idx]

                confidence = probabilities[
                    pred_idx
                ].item()

                new_detections.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        label,
                        confidence
                    )
                )

            self.last_detections = new_detections

        for (
            x,
            y,
            w,
            h,
            label,
            confidence
        ) in self.last_detections:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            display_text = (
                f"{label.upper()}: "
                f"{confidence * 100:.1f}%"
            )

            text_top = max(0, y - 35)

            cv2.rectangle(
                image,
                (x, text_top),
                (x + w, y),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                image,
                display_text,
                (x + 5, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


st.title("😊 Live Facial Expression Recognition")

st.write(
    "Turn on your webcam and detect faces and emotions "
    "continuously from the live video stream."
)

st.info(
    "Click START below and allow camera access when your browser "
    "asks for permission."
)

webrtc_ctx = webrtc_streamer(
    key="live-emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15},
        },
        "audio": False,
    },
    video_processor_factory=EmotionVideoProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("🟢 Live camera is running — detecting emotions.")
else:
    st.warning(
        "🔴 Camera is stopped. Click START to begin live detection."
    )

with st.sidebar:

    st.header("About")

    st.write(
        "Real-time facial expression recognition using:"
    )

    st.write("• OpenCV Haar Cascade — face detection")
    st.write("• Vision Transformer (ViT) — emotion recognition")
    st.write("• WebRTC — live browser camera streaming")

    st.write("### Model")
    st.code(MODEL_NAME)

    st.write("### Device")
    st.write(str(device))

    st.write("### Processing")
    st.write(
        "ViT inference runs every 6th frame to reduce CPU load "
        "while keeping the video stream live."
    )
