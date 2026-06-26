import cv2
import time
import json
import socket
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "./models/yolov8n_openvino_model"

CAMERA_ID = 0
WIDTH = 640
HEIGHT = 480
FPS = 30

DISPLAY_LABEL = "Container"

# YOLO classes accepted as possible containers.
TARGET_CLASSES = {
    "mouse",
    "cell phone",
    "remote",
    "bottle",
    "sports ball",
    "book",
    "skateboard",
    "hot_dog",
    "carrot"
}
# Maximum number of objects to show and send.
MAX_OBJECTS = 2

# Confidence is used internally only.
# It is not displayed and not sent in the JSON.
CONF_THRESHOLD = 0.05

# Size filter.
# If the bounding box is too large, ignore it.
MAX_BOX_AREA_RATIO = 0.12
MAX_BOX_WIDTH_RATIO = 0.55
MAX_BOX_HEIGHT_RATIO = 0.55

IMGSZ = 320
JPEG_QUALITY = 55
PRINT_EVERY_SECONDS = 1.0

# -----------------------------
# UDP settings
# -----------------------------

ENABLE_UDP = True

# Robot workstation address.
# ROBOT_PC_IP = "192.168.11.0"
ROBOT_PC_IP = "localhost"

UDP_PORT = 5555
SEND_EVERY_SECONDS = 0.2

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_send_time = 0.0

print(f"Sending UDP target messages to {ROBOT_PC_IP}:{UDP_PORT}")

# -----------------------------
# Load OpenVINO model
# -----------------------------

print(f"Loading {MODEL_PATH} for OpenVINO inference...")
model = YOLO(MODEL_PATH, task="detect")

# -----------------------------
# Open camera
# -----------------------------

cap = cv2.VideoCapture("/dev/global_camera", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera /dev/video1")

app = Flask(__name__)

latest_targets = {
    "valid": False,
    "count": 0,
    "label": DISPLAY_LABEL,
    "targets": [],
    "message": "No container detected",
    "timestamp": time.time()
}


def is_box_too_big(x1, y1, x2, y2):
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    frame_area = WIDTH * HEIGHT

    area_ratio = box_area / frame_area
    width_ratio = box_width / WIDTH
    height_ratio = box_height / HEIGHT

    if area_ratio > MAX_BOX_AREA_RATIO:
        return True

    if width_ratio > MAX_BOX_WIDTH_RATIO:
        return True

    if height_ratio > MAX_BOX_HEIGHT_RATIO:
        return True

    return False


def extract_targets(result):
    internal_targets = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        detected_class = model.names[cls_id]

        if detected_class not in TARGET_CLASSES:
            continue

        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if is_box_too_big(x1, y1, x2, y2):
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        area = int((x2 - x1) * (y2 - y1))

        target = {
            "_score": score,
            "label": DISPLAY_LABEL,
            "center_px": [cx, cy],
            "bbox_px": [x1, y1, x2, y2],
            "detected_as": detected_class,
            "image_size": [WIDTH, HEIGHT],
            "area_px": area,
            "timestamp": time.time()
        }

        internal_targets.append(target)

    # Keep the two strongest valid detections.
    internal_targets.sort(key=lambda t: t["_score"], reverse=True)
    internal_targets = internal_targets[:MAX_OBJECTS]

    # Remove internal confidence score before display and before UDP output.
    visible_targets = []

    for target in internal_targets:
        clean_target = dict(target)
        clean_target.pop("_score", None)
        visible_targets.append(clean_target)

    return visible_targets


def draw_targets(frame, targets):
    if len(targets) == 0:
        cv2.putText(
            frame,
            "No container detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )
        return frame

    for i, target in enumerate(targets, start=1):
        x1, y1, x2, y2 = target["bbox_px"]
        cx, cy = target["center_px"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

        main_label = f"{DISPLAY_LABEL} {i} center=({cx}, {cy})"

        cv2.putText(
            frame,
            main_label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    summary_parts = []

    for i, target in enumerate(targets, start=1):
        cx, cy = target["center_px"]
        summary_parts.append(f"{i}:({cx},{cy})")

    summary = " | ".join(summary_parts)

    cv2.putText(
        frame,
        f"Containers: {len(targets)}  Centers: {summary}",
        (20, HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame


def send_targets_udp(message, now):
    global last_send_time

    if not ENABLE_UDP:
        return

    if now - last_send_time >= SEND_EVERY_SECONDS:
        data = json.dumps(message).encode("utf-8")
        udp_socket.sendto(data, (ROBOT_PC_IP, UDP_PORT))
        last_send_time = now


def generate_frames():
    global latest_targets

    last_print_time = 0.0
    frame_counter = 0
    fps_start_time = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        frame_counter += 1

        result = model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False
        )[0]

        targets = extract_targets(result)

        now = time.time()

        latest_targets = {
            "valid": len(targets) > 0,
            "count": len(targets),
            "label": DISPLAY_LABEL,
            "targets": targets,
            "message": "Container detected" if len(targets) > 0 else "No container detected",
            "timestamp": now
        }

        send_targets_udp(latest_targets, now)

        frame = draw_targets(frame, targets)

        if now - last_print_time >= PRINT_EVERY_SECONDS:
            elapsed = now - fps_start_time
            measured_fps = frame_counter / elapsed if elapsed > 0 else 0.0

            if len(targets) > 0:
                print("FPS:", round(measured_fps, 2))
                print("UDP SENT TO:", f"{ROBOT_PC_IP}:{UDP_PORT}")
                print("TARGETS:", json.dumps(latest_targets, indent=2))
            else:
                print("FPS:", round(measured_fps, 2), "| No container detected")
                print("UDP SENT TO:", f"{ROBOT_PC_IP}:{UDP_PORT}")

            last_print_time = now

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>OpenVINO Container Detector</title>
      </head>
      <body style="background:#111;color:white;font-family:sans-serif;">
        <h2>OpenVINO Container Detector</h2>
        <p>Running on Intel Pantherlake with OpenVINO inference.</p>
        <p>Sending target JSON to robot workstation using UDP socket.</p>
        <p>Robot workstation target: 192.168.11.0:5555.</p>
        <img src="/video" width="640">
        <p>Latest target JSON: <a href="/targets" style="color:#00ff88;">/targets</a></p>
      </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/targets")
def targets():
    return jsonify(latest_targets)


if __name__ == "__main__":
    print("Running OpenVINO container detector on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
