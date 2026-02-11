import threading
import time
import json
import numpy as np
import cv2
import torch
from datetime import datetime
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import pycozmo


class CozmoObserver:
    def __init__(self, emotion_model_path: str, yolo_model_path: str):
        # State control
        self.running = False
        self.lock = threading.Lock()
        self.start_time = None

        # Data buffers
        self.latest_raw_frame = None
        self._raw_face_box = None       # [x1, y1, x2, y2] in 640×480 space
        self._latest_emotion = "none"
        self._emotion_conf = 0.0

        # AI Models
        print("[Observer] Loading AI Models...")
        self.processor = AutoImageProcessor.from_pretrained(emotion_model_path, local_files_only=True)
        self.emotion_model = AutoModelForImageClassification.from_pretrained(
            emotion_model_path, local_files_only=True
        ).to("cpu")
        self.face_model_yolo = YOLO(yolo_model_path)

    # ────────────────────────────────────────────────
    #  ZONE 1: RAW DATA COLLECTION
    # ────────────────────────────────────────────────
    def on_camera_image(self, cli, new_im):
        """Callback from Cozmo's camera."""
        with self.lock:
            self.latest_raw_frame = np.array(new_im)

    def _analysis_loop(self):
        """Background thread: run detection as fast as possible."""
        while self.running:
            frame = None
            with self.lock:
                if self.latest_raw_frame is not None:
                    frame = self.latest_raw_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # 1. Face Detection (YOLO)
            res = self.face_model_yolo(
                cv2.resize(frame, (640, 480)), conf=0.3, verbose=False
            )[0]
            boxes = res.boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                # Pick the largest face
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_box = boxes[np.argmax(areas)]

                with self.lock:
                    self._raw_face_box = best_box.tolist()  # store as python list

                # 2. Emotion Recognition
                # Scale box coordinates back to original frame size (assuming 320×240 crop target)
                x1, y1, x2, y2 = (best_box * [320/640, 240/480, 320/640, 240/480]).astype(int)
                face_roi = frame[max(0, y1):min(frame.shape[0], y2),
                                 max(0, x1):min(frame.shape[1], x2)]

                if face_roi.size > 0:
                    inputs = self.processor(images=PILImage.fromarray(face_roi), return_tensors="pt")
                    with torch.no_grad():
                        out = self.emotion_model(**inputs)
                        probs = torch.nn.functional.softmax(out.logits, dim=-1)
                        conf, idx = torch.max(probs, dim=-1)

                        label = self.emotion_model.config.id2label[idx.item()].lower()
                        with self.lock:
                            self._latest_emotion = label
                            self._emotion_conf = float(round(conf.item(), 3))
            else:
                with self.lock:
                    self._raw_face_box = None
                    self._latest_emotion = "none"
                    self._emotion_conf = 0.0

            time.sleep(0.01)

    # ────────────────────────────────────────────────
    #  ZONE 2: LOGIC & MAPPING
    # ────────────────────────────────────────────────
    def get_raw_status(self):
        with self.lock:
            return {
                "box": self._raw_face_box,
                "emotion": self._latest_emotion,
                "confidence": self._emotion_conf
            }

    def _map_mood_metrics(self, label: str, confidence: float):
        mapping = {
            "happy":   {"val": 0.85, "att": 0.80, "smile": 0.90},
            "neutral": {"val": 0.50, "att": 0.60, "smile": 0.05},
            "sad":     {"val": 0.20, "att": 0.30, "smile": 0.00},
            "angry":   {"val": 0.15, "att": 0.95, "smile": 0.00},
            "none":    {"val": 0.00, "att": 0.00, "smile": 0.00}
        }
        m = mapping.get(label, mapping["neutral"])
        return {
            "valence":   {"value": m["val"], "confidence": confidence},
            "attention": {"value": m["att"], "confidence": min(1.0, confidence + 0.1)},
            "smile":     {"value": m["smile"], "confidence": confidence}
        }

    def describe_position(self, cx: float, cy: float, dist: float) -> dict:
        # Horizontal
        if cx < 0.35:
            h_pos = "left"
        elif cx > 0.65:
            h_pos = "right"
        else:
            h_pos = "center"

        # Vertical
        if cy < 0.35:
            v_pos = "top"
        elif cy > 0.65:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Combined natural language
        if h_pos == "center" and v_pos == "middle":
            where = "perfectly in the center"
        elif h_pos == "center":
            where = f"in the {v_pos} center"
        elif v_pos == "middle":
            where = f"in the {h_pos} middle"
        else:
            where = f"to the {v_pos} {h_pos}"

        # Distance interpretation (tune thresholds to your liking)
        if dist < 0.8:
            distance_word = "very close"
        elif dist < 1.4:
            distance_word = "close"
        elif dist < 2.5:
            distance_word = "at medium distance"
        else:
            distance_word = "far away"

        return {
            "horizontal": h_pos,
            "vertical": v_pos,
            "where": where,
            "distance": distance_word,
            "natural": f"The person is {where}, {distance_word}."
        }

    # ────────────────────────────────────────────────
    #  ZONE 3: EXPORTING (JSON)
    # ────────────────────────────────────────────────
    def get_human_presence_only(self):
        with self.lock:
            detected = self._raw_face_box is not None

        return {
            "timestamp": datetime.now().isoformat(),
            "human_detected": detected
        }

    def get_event_agnostic_metrics(self):
        state = self.get_raw_status()

        human_present = state["box"] is not None
        pos = [0.0, 0.0, 0.0]
        cx = cy = dist = 0.0
        zone = 0
        position_desc = {
            "horizontal": "none",
            "vertical": "none",
            "where": "no person detected",
            "distance": "—",
            "natural": "No person is currently visible."
        }

        if human_present and state["box"] is not None:
            # Convert numpy → python floats
            box = [float(v) for v in state["box"]]

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / 640
            cy = (y1 + y2) / 2 / 480
            height = y2 - y1

            dist = 150.0 / max(height, 0.1)
            dist = round(dist, 2)

            pos = [round(cx, 3), round(cy, 3), dist]
            zone = 1 if dist < 1.3 else 2

            position_desc = self.describe_position(cx, cy, dist)

        mood = self._map_mood_metrics(state["emotion"], float(state["confidence"]))

        return {
            "timestamp": datetime.now().isoformat(),
            "human_presence_simple": {
                "detected": human_present
            },
            "event_agnostic": {
                "human_presence": {
                    "human_present": human_present,
                    "person_position": pos,
                    "position_description": position_desc,
                    "engagement_zone": zone
                },
                "emotion_recognition": {
                    "present": human_present,
                    "mood": mood if human_present else {
                        "valence":   {"value": 0.0, "confidence": 0.0},
                        "attention": {"value": 0.0, "confidence": 0.0},
                        "smile":     {"value": 0.0, "confidence": 0.0}
                    }
                }
            }
        }

    def start(self):
        self.running = True
        self.start_time = time.time()
        threading.Thread(target=self._analysis_loop, daemon=True).start()


# ────────────────────────────────────────────────
#  MAIN EXECUTION
# ────────────────────────────────────────────────
if __name__ == "__main__":
    MODELS = {
        "emo": "models/emotion_convnext",
        "yolo": "models/yolo_face/model.pt"
    }

    try:
        with pycozmo.connect(enable_procedural_face=False) as cli:
            cli.set_head_angle(0.6)
            cli.enable_camera(color=True)

            observer = CozmoObserver(MODELS["emo"], MODELS["yolo"])
            cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer.on_camera_image)
            observer.start()

            print("Cozmo Observer started. Press Ctrl+C to stop.\n")

            while True:
                metrics = observer.get_event_agnostic_metrics()
                print(json.dumps(metrics, indent=2))
                print("-" * 70)
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    except Exception as e:
        print(f"Error: {e}")