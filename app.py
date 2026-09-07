import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# =============================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="wide"
)

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"


# =============================================================
# 1. LOAD EMOTION RECOGNITION MODEL
# =============================================================

@st.cache_resource
def load_emotion_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return processor, model, device


# =============================================================
# 2. LOAD FACE DETECTION CASCADE
# =============================================================

@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade.")

    return face_cascade


# =============================================================
# 3. FACE + EMOTION DETECTION
# =============================================================

def detect_emotions(image, processor, model, device, face_cascade):
    # PIL RGB -> OpenCV BGR
    rgb_image = np.array(image)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    detections = []

    for (x, y, w, h) in faces:

        # Crop face
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size == 0:
            continue

        # BGR -> RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        pil_face = Image.fromarray(face_rgb)

        # Preprocess
        inputs = processor(
            images=pil_face,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ViT inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Calculate probabilities
        probabilities = F.softmax(logits, dim=-1)[0]

        pred_idx = torch.argmax(probabilities).item()

        label = model.config.id2label[pred_idx]

        confidence = probabilities[pred_idx].item()

        detections.append(
            (x, y, w, h, label, confidence)
        )

    # Draw results
    for (
        x,
        y,
        w,
        h,
        label,
        confidence
    ) in detections:

        # Bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Emotion text
        display_text = (
            f"{label.upper()}: "
            f"{confidence * 100:.1f}%"
        )

        # Text background
        text_y1 = max(0, y - 35)

        cv2.rectangle(
            frame,
            (x, text_y1),
            (x + w, y),
            (0, 255, 0),
            cv2.FILLED
        )

        # Text
        cv2.putText(
            frame,
            display_text,
            (x + 5, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # OpenCV BGR -> RGB for Streamlit
    result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return result_image, detections


# =============================================================
# 4. STREAMLIT INTERFACE
# =============================================================

st.title("😊 Facial Expression Recognition")

st.markdown(
    "Detect facial expressions using a **ViT emotion recognition model** "
    "and OpenCV face detection."
)

st.info(
    "For Streamlit deployment, the camera is accessed through the browser. "
    "The server cannot use cv2.VideoCapture(0) to access your computer's webcam."
)

# Load models
with st.spinner("Loading AI model..."):
    processor, model, device = load_emotion_model()
    face_cascade = load_face_detector()

st.success(f"Model loaded successfully on: {device}")

# =============================================================
# 5. CAMERA INPUT
# =============================================================

st.subheader("📷 Camera")

camera_image = st.camera_input("Take a picture")

# =============================================================
# 6. IMAGE UPLOAD
# =============================================================

st.subheader("🖼️ Or upload an image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

image_source = camera_image if camera_image is not None else uploaded_image

# =============================================================
# 7. PROCESS IMAGE
# =============================================================

if image_source is not None:

    image = Image.open(image_source).convert("RGB")

    with st.spinner("Detecting face and emotion..."):
        result_image, detections = detect_emotions(
            image,
            processor,
            model,
            device,
            face_cascade
        )

    st.subheader("🎯 Detection Result")

    st.image(
        result_image,
        caption="Facial Expression Detection",
        use_container_width=True
    )

    # =========================================================
    # DETECTION RESULTS
    # =========================================================

    if detections:

        st.subheader("📊 Results")

        for i, (
            x,
            y,
            w,
            h,
            label,
            confidence
        ) in enumerate(detections, start=1):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Face", i)

            with col2:
                st.metric("Emotion", label.upper())

            with col3:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

    else:
        st.warning(
            "No face detected. Please try another image with a clear face."
        )

else:
    st.write(
        "📷 Take a picture with your camera or upload an image to begin."
    )

# =============================================================
# 8. SIDEBAR
# =============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses a Vision Transformer (ViT) model "
        "for facial expression recognition."
    )

    st.write("**Model:**")
    st.code(MODEL_NAME)

    st.write("**Face Detector:**")
    st.write("OpenCV Haar Cascade")

    st.write("**Device:**")
    st.write(str(device))
