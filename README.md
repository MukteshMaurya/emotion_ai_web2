# AI Facial Expression Recognition Web App

This version uses the camera of the device that opens the web page.

## Architecture

Device camera -> Browser JavaScript -> Flask `/predict` -> Haar face detection -> ViT -> JSON -> Browser overlay

The backend does NOT use `cv2.VideoCapture(0)`.

## Project structure

```text
emotion_ai_web/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The browser will ask for camera permission.

## Test from another device on the same Wi-Fi

Run the app:

```bash
python app.py
```

Find the computer's IPv4 address with:

```powershell
ipconfig
```

Then open on another device:

```text
http://YOUR_COMPUTER_IP:5000
```

This is only for LAN testing.

## Public deployment

For a public URL, deploy this application to a cloud host.

Important: browser camera access requires a secure context. A public deployment should therefore use HTTPS.

Set the cloud platform's PORT environment variable if it provides one. `app.py` already reads it.

For a production Linux deployment using Gunicorn:

```bash
gunicorn app:app
```

## Important

The model runs on the server. Every visitor's browser uses that visitor's own camera.

For example:

Phone -> phone camera -> server -> ViT -> phone

Laptop -> laptop camera -> server -> ViT -> laptop

Multiple visitors can use the site, but each prediction consumes server CPU/GPU resources.

## Performance

The browser displays the camera directly, while AI requests are sent approximately every 200 ms (about 5 predictions/sec). This prevents slow ViT inference from making the camera preview itself lag.

The backend also batches multiple detected faces into one ViT call.
