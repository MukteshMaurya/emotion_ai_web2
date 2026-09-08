import threading
import time

import av
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from twilio.rest import Client
from transformers import ViTForImageClassification, ViTImageProcessor


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Live Facial Expression Recognition",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"


# ============================================================
# TWILIO ICE SERVERS
# ============================================================

@st.cache_data(ttl=3000, show_spinner=False)
def get_ice_servers(account_sid, auth_token):

    client = Client(
        account_sid,
        auth_token
    )

    token = client.tokens.create(
        ttl=3600
    )

    return token.ice_servers


def create_rtc_configuration():

    try:

        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

    except KeyError:

        st.error(
            "Twilio credentials are missing. "
            "Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN "
            "in Streamlit Cloud → Settings → Secrets."
        )

        st.stop()

    try:

        ice_servers = get_ice_servers(
            account_sid,
            auth_token
        )

        return {
            "iceServers": ice_servers
        }

    except Exception as e:

        st.error(
            "Could not create the WebRTC ICE configuration."
        )

        st.exception(e)

        st.stop()


rtc_configuration = create_rtc_configuration()


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    processor = ViTImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    model.eval()

    return processor, model, device


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    if detector.empty():

        raise RuntimeError(
            "OpenCV Haar Cascade could not be loaded."
        )

    return detector


# ============================================================
# LOAD AI
# ============================================================

with st.spinner("Loading AI model..."):

    processor, model, device = load_emotion_model()

    face_cascade = load_face_detector()


# ============================================================
# MODEL LOCK
# ============================================================

model_lock = threading.Lock()


# ============================================================
# CALLBACK STATE
# ============================================================

frame_counter = 0

last_prediction = None

last_confidence = 0.0

last_box = None

last_prediction_time = 0


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# AI runs only once every N frames.
#
# Increase this value for more speed.
#
# 1 = every frame
# 3 = every 3rd frame
# 5 = every 5th frame
# 8 = every 8th frame
#
AI_FRAME_SKIP = 5


# Face detection also doesn't need to run
# on every frame.

FACE_DETECTION_SKIP = 3


# Maximum size used for AI processing.

MAX_FACE_SIZE = 224


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(face_bgr):

    if face_bgr is None:
        return None, 0.0

    if face_bgr.size == 0:
        return None, 0.0

    # Resize face before sending to ViT.
    face_bgr = cv2.resize(
        face_bgr,
        (MAX_FACE_SIZE, MAX_FACE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # BGR → RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    # OpenCV image → PIL
    pil_face = Image.fromarray(
        face_rgb
    )

    # Prepare model input
    inputs = processor(
        images=pil_face,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # AI inference
    with model_lock:

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

    logits = outputs.logits

    probabilities = F.softmax(
        logits,
        dim=-1
    )[0]

    pred_idx = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        pred_idx
    ].item()

    label = model.config.id2label[
        pred_idx
    ]

    return label, confidence


# ============================================================
# DRAW RESULT
# ============================================================

def draw_prediction(
    frame,
    box,
    label,
    confidence
):

    if box is None:
        return frame

    x, y, w, h = box

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    if label is None:
        return frame

    text = (
        f"{label.upper()}: "
        f"{confidence * 100:.1f}%"
    )

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    text_top = max(
        0,
        y - text_height - 12
    )

    # Background
    cv2.rectangle(
        frame,
        (x, text_top),
        (
            x + text_width + 10,
            y
        ),
        (0, 255, 0),
        cv2.FILLED
    )

    # Text
    cv2.putText(
        frame,
        text,
        (
            x + 5,
            y - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# WEBRTC VIDEO CALLBACK
# ============================================================

def video_frame_callback(
    frame: av.VideoFrame
) -> av.VideoFrame:

    global frame_counter
    global last_prediction
    global last_confidence
    global last_box
    global last_prediction_time

    # Convert WebRTC frame to OpenCV
    image = frame.to_ndarray(
        format="bgr24"
    )

    frame_counter += 1

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    if (
        frame_counter % FACE_DETECTION_SKIP == 0
        or last_box is None
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:

            # Select largest face
            largest_face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            last_box = largest_face

        else:

            last_box = None

            last_prediction = None

            last_confidence = 0.0


    # --------------------------------------------------------
    # EMOTION DETECTION
    # --------------------------------------------------------

    if last_box is not None:

        x, y, w, h = last_box

        face_roi = image[
            y:y + h,
            x:x + w
        ]

        # Run AI only every few frames
        if (
            frame_counter % AI_FRAME_SKIP == 0
            or last_prediction is None
        ):

            try:

                label, confidence = predict_emotion(
                    face_roi
                )

                last_prediction = label

                last_confidence = confidence

                last_prediction_time = time.time()

            except Exception:

                pass


        # ----------------------------------------------------
        # DRAW RESULT
        # ----------------------------------------------------

        image = draw_prediction(
            image,
            last_box,
            last_prediction,
            last_confidence
        )


    # --------------------------------------------------------
    # RETURN FRAME
    # --------------------------------------------------------

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "😊 Live Facial Expression Recognition"
)


st.write(
    "Real-time facial expression detection "
    "using OpenCV and a Vision Transformer (ViT)."
)


st.info(
    "Click START and allow camera permission "
    "to begin live emotion detection."
)


# ============================================================
# WEBRTC
# ============================================================

webrtc_ctx = webrtc_streamer(

    key="live-facial-expression",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=rtc_configuration,

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640
            },
            "height": {
                "ideal": 480
            },
            "frameRate": {
                "ideal": 10
            }
        },
        "audio": False
    },

    async_processing=True
)


# ============================================================
# STATUS
# ============================================================

if webrtc_ctx.state.playing:

    st.success(
        "🟢 Camera connected — "
        "live emotion detection is running."
    )

else:

    st.warning(
        "🔴 Camera is stopped. "
        "Click START above."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "Live facial expression recognition "
        "from the browser webcam."
    )

    st.write(
        "**Face detection:** "
        "OpenCV Haar Cascade"
    )

    st.write(
        "**Emotion recognition:** "
        "ViT"
    )

    st.write(
        "**Live video:** "
        "WebRTC"
    )

    st.write(
        "**Network traversal:** "
        "Twilio TURN/STUN"
    )

    st.divider()

    st.write("**Model**")

    st.code(
        MODEL_NAME
    )

    st.write("**Device**")

    st.code(
        str(device)
    )

    st.divider()

    st.write(
        "**Performance**"
    )

    st.write(
        f"Emotion inference: "
        f"every {AI_FRAME_SKIP} frames"
    )

    st.write(
        f"Face detection: "
        f"every {FACE_DETECTION_SKIP} frames"
    )

    st.divider()

    st.caption(
        "The browser webcam is accessed "
        "through WebRTC. cv2.VideoCapture(0) "
        "is not used."
    )
