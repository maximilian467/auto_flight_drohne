import cv2
import numpy as np
import time
from ultralytics import YOLO
from picamera2 import Picamera2

print("Lade KI-Gehirn (YOLOv8)...")
model = YOLO('yolov8n.pt')

print("Initialisiere Raspberry Pi Camera...")
picam2 = Picamera2()

# Performance-Optimierung: Kleine Auflösung
CAM_WIDTH = 320
CAM_HEIGHT = 240
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

camera_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (CAM_WIDTH, CAM_HEIGHT)},
    controls={"FrameDurationLimits": (16666, 16666)}  # ~60 FPS Kamera
)
picam2.configure(camera_config)
picam2.set_controls({"ExposureTime": 5000, "AnalogueGain": 2.0})
picam2.start()
time.sleep(0.3)

cv2.namedWindow("Bench Test HUD", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Bench Test HUD", cv2.WND_PROP_TOPMOST, 1)

# Tracking-Variablen
TARGET_CLASS = 2  # Auto
MIN_LOCK_FRAMES = 15
LOST_TIMEOUT = 60  # ~1 Sekunde bei 60 FPS

pred_x = 0
pred_y = 0
pred_w = 0.0
vel_x = 0.0
vel_w = 0.0
locked_vel_x = 0.0
locked_vel_w = 0.0

frames_mit_ziel = 0
target_locked = False
target_tracking = False  # True = wir verfolgen aktiv (auch bei Verlust)
frames_ohne_sicht = 0

# YOLO Klassennamen von COCO-Datensatz
CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

# Farben für häufige Klassen (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),    # person - Blau
    1: (0, 255, 255),  # bicycle - Gelb
    2: (0, 255, 0),    # car - Grün
    3: (0, 0, 255),    # motorcycle - Rot
    5: (255, 0, 255),  # bus - Magenta
    7: (0, 128, 255),  # truck - Orange
}

start_time = time.time()
fps_counter = 0
last_fps_time = time.time()
current_fps = 0
frame_skip = 3  # YOLO nur jeden 3. Frame
frame_count = 0
last_results = None

print("System online! Starte...")
print("Ziel: Erst nach LOST_TIMEOUT Frames Tracking abbrechen")

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    # Display upscale
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cam_height, cam_width = display_frame.shape[:2]
    center_x = cam_width // 2

    # YOLO nur alle N Frames
    frame_count += 1
    if frame_count % frame_skip == 0:
        results = model(frame, verbose=False, imgsz=320)
        last_results = results

    # Verarbeite letzte Ergebnisse
    auto_gesehen = False
    target_det = None

    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Koordinaten skalieren
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                scale_x = DISPLAY_WIDTH / CAM_WIDTH
                scale_y = DISPLAY_HEIGHT / CAM_HEIGHT

                x1d = int(x1 * scale_x)
                y1d = int(y1 * scale_y)
                x2d = int(x2 * scale_x)
                y2d = int(y2 * scale_y)
                cxd = int(((x1 + x2) / 2) * scale_x)
                cyd = int(((y1 + y2) / 2) * scale_y)

                # Farbe wählen
                color = CLASS_COLORS.get(cls_id, (128, 128, 128))
                name = CLASS_NAMES.get(cls_id, f"class{cls_id}")

                # Bounding Box zeichnen
                cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), color, 2)
                cv2.circle(display_frame, (cxd, cyd), 5, color, -1)

                # Label
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (x1d, y1d - th - 8), (x1d + tw, y1d), color, -1)
                cv2.putText(display_frame, label, (x1d, y1d - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Tracking nur für Autos (Klasse 2)
                if cls_id == TARGET_CLASS and target_det is None and conf > 0.5:
                    target_det = {
                        'x': cxd,
                        'y': cyd,
                        'w': x2d - x1d
                    }
                    # Gelber Ring für Target
                    cv2.circle(display_frame, (cxd, cyd), 10, (0, 255, 255), 2)

    # ------------------------------------------
    # TARGET TRACKING LOGIK
    # ------------------------------------------

    if target_det:
        # Ziel gesehen - Tracking-Daten aktualisieren
        current_x = target_det['x']
        current_y = target_det['y']
        current_w = target_det['w']

        vel_x = 0.3 * (current_x - pred_x) + 0.7 * vel_x
        vel_w = 0.5 * (current_w - pred_w) + 0.5 * vel_w

        pred_x, pred_y, pred_w = current_x, current_y, current_w
        locked_vel_x, locked_vel_w = vel_x, vel_w

        frames_mit_ziel += 1
        frames_ohne_sicht = 0
        auto_gesehen = True

        # Target Locked erst nach MIN_LOCK_FRAMES
        if frames_mit_ziel >= MIN_LOCK_FRAMES:
            target_locked = True
            target_tracking = True

    else:
        # Ziel nicht gesehen
        frames_ohne_sicht += 1

        # Während des Trackings: Prädiktion fortsetzen
        if target_tracking:
            if frames_ohne_sicht < LOST_TIMEOUT:
                # Prädiktion basierend auf letzter Geschwindigkeit
                pred_x += locked_vel_x
                pred_w += locked_vel_w

                # Begrenze Prädiktion auf Bildbereich mit Puffer
                pred_x = max(-100, min(cam_width + 100, pred_x))
                pred_w = max(10, pred_w)

                # Zeichne vorhergesagte Position
                cv2.circle(display_frame, (int(pred_x), int(pred_y)), 12, (255, 0, 0), 2)
                cv2.circle(display_frame, (int(pred_x), int(pred_y)), 5, (255, 0, 0), -1)
                cv2.putText(display_frame, "PREDICTED", (int(pred_x) - 40, int(pred_y) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            else:
                # Timeout erreicht - Tracking abbrechen
                if target_locked:
                    print(f"[{time.time()-start_time:.1f}s] TARGET LOST - Tracking abgebrochen")
                target_locked = False
                target_tracking = False
                frames_mit_ziel = 0
                pred_x = 0
                pred_y = 0
                pred_w = 0.0
                last_results = None

    # FPS berechnen
    fps_counter += 1
    if time.time() - last_fps_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        last_fps_time = time.time()

    cv2.putText(display_frame, f"FPS: {current_fps}", (DISPLAY_WIDTH - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ------------------------------------------
    # VISUALISIERUNG & STEUERUNG
    # ------------------------------------------

    if target_tracking:
        if target_locked:
            # Grüner Rahmen = gelocked
            cv2.rectangle(display_frame, (5, 5), (cam_width - 5, cam_height - 5), (0, 255, 0), 3)
            cv2.putText(display_frame, "TARGET LOCKED", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Steuerung berechnen
            if pred_w > 0:
                future_target_x = pred_x + (locked_vel_x * 3)
                pixel_fehler_x = center_x - future_target_x
                yaw_rate = np.clip(pixel_fehler_x * 0.006, -0.85, 0.85)

                steuerung_text = f"YAW: {yaw_rate:.2f}"
                color = (0, 0, 255) if abs(yaw_rate) > 0.1 else (0, 255, 0)
                if yaw_rate > 0.1:
                    steuerung_text += " < LEFT"
                elif yaw_rate < -0.1:
                    steuerung_text += " > RIGHT"

                cv2.putText(display_frame, steuerung_text, (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Zeichne Ziel-Linie
                cv2.line(display_frame, (int(pred_x), cam_height // 2), (center_x, cam_height // 2), (0, 255, 255), 2)
        else:
            # Gelber Rahmen = acquiring
            cv2.rectangle(display_frame, (5, 5), (cam_width - 5, cam_height - 5), (0, 255, 255), 2)
            progress = min(frames_mit_ziel, MIN_LOCK_FRAMES) / MIN_LOCK_FRAMES * 100
            cv2.putText(display_frame, f"ACQUIRING... {progress:.0f}%", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Frames: {frames_mit_ziel}/{MIN_LOCK_FRAMES}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Statusanzeige für Tracking
        status_text = f"Tracking | Lost: {frames_ohne_sicht}/{LOST_TIMEOUT}"
        cv2.putText(display_frame, status_text, (20, cam_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    else:
        # Roter Rahmen = suche
        cv2.rectangle(display_frame, (5, 5), (cam_width - 5, cam_height - 5), (0, 0, 255), 2)
        cv2.putText(display_frame, "SEARCHING...", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Bench Test HUD", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
