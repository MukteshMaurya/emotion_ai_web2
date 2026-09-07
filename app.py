import os
import io
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
from transformers import ViTImageProcessor, ViTForImageClassification


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

MAX_IMAGE_WIDTH = 640

# Minimum time between AI predictions from the browser.
# 0.20 = approximately 5 predictions per second.
INFERENCE_INTERVAL = 0.20


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

print("Loading ViT emotion model...")

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

model = ViTForImageClassification.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")


# ============================================================
# LOAD HAAR FACE DETECTOR
# ============================================================

CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "4.x/data/haarcascades/haarcascade_frontalface_default.xml"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_FILE = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)


if not os.path.exists(CASCADE_FILE):
    print("Downloading face detector XML...")

    urllib.request.urlretrieve(
        CASCADE_URL,
        CASCADE_FILE
    )


face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar face detector.")

print("Face detector loaded successfully.")


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotions(face_images):
    """
    Predict emotions for multiple face images in one batch.
    """

    if not face_images:
        return []

    pil_images = []

    for face in face_images:

        rgb_face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb_face)

        pil_images.append(pil_image)

    inputs = processor(
        images=pil_images,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        confidences, class_ids = torch.max(
            probabilities,
            dim=-1
        )

    results = []

    for confidence, class_id in zip(
        confidences,
        class_ids
    ):

        emotion = model.config.id2label[
            int(class_id)
        ]

        confidence_value = float(
            confidence.item()
        )

        results.append({
            "emotion": emotion,
            "confidence": confidence_value
        })

    return results


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "frame" not in request.files:
        return jsonify({
            "error": "No frame received."
        }), 400

    try:

        # ----------------------------------------------------
        # Read uploaded JPEG frame
        # ----------------------------------------------------

        image_bytes = request.files["frame"].read()

        if not image_bytes:
            return jsonify({
                "error": "Empty frame."
            }), 400

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Could not decode image."
            }), 400


        # ----------------------------------------------------
        # Resize large images
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        if width > MAX_IMAGE_WIDTH:

            scale = MAX_IMAGE_WIDTH / width

            new_width = MAX_IMAGE_WIDTH
            new_height = int(height * scale)

            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )


        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect on a smaller image for speed.
        small_gray = cv2.resize(
            gray,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        detected_faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )


        # Convert coordinates back to original image size.
        faces = []

        for x, y, w, h in detected_faces:

            faces.append((
                int(x * 2),
                int(y * 2),
                int(w * 2),
                int(h * 2)
            ))


        # ----------------------------------------------------
        # Prepare face crops
        # ----------------------------------------------------

        face_crops = []

        for x, y, w, h in faces:

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                frame.shape[1],
                x + w
            )

            y2 = min(
                frame.shape[0],
                y + h
            )

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            face_crops.append(face)


        # ----------------------------------------------------
        # Predict emotions
        # ----------------------------------------------------

        predictions = predict_emotions(
            face_crops
        )


        # ----------------------------------------------------
        # Build response
        # ----------------------------------------------------

        response_faces = []

        prediction_index = 0

        for x, y, w, h in faces:

            if prediction_index >= len(predictions):
                break

            prediction = predictions[
                prediction_index
            ]

            response_faces.append({

                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "emotion": prediction[
                    "emotion"
                ],

                "confidence": prediction[
                    "confidence"
                ]
            })

            prediction_index += 1


        return jsonify({

            "faces": response_faces,

            "device": str(DEVICE)
        })


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )