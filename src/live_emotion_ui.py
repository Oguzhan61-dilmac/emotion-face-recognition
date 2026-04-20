import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

MODELS_DIR = "models"
PROTO_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
EMO_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn.h5")

EMOTIONS = ["Kizgin", "Igrenme", "Korku", "Mutlu", "Uzgun", "Saskin", "Notr"]

FACE_CONF_THRESHOLD = 0.60
EMA_ALPHA_DEFAULT = 0.35
PANEL_WIDTH = 430
TIMELINE_SECONDS = 30
TIMELINE_SAMPLES = 150
ACCENT = (0, 196, 255)
TEXT_MAIN = (232, 236, 242)
TEXT_MUTED = (154, 164, 178)
PANEL_BG = (14, 18, 24)
CARD_BG = (24, 29, 36)


class SessionLogger:
    def __init__(self, path: Path, emotion_labels: Sequence[str]):
        self.path = path
        self.emotion_labels = list(emotion_labels)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "dominant", *self.emotion_labels])

    def log(self, timestamp_text: str, dominant: str, probs: np.ndarray) -> None:
        row = [timestamp_text, dominant, *[f"{float(p):.6f}" for p in probs]]
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()


class VideoRecorder:
    def __init__(self, writer: cv2.VideoWriter, path: Path, codec: str, fps: float):
        self.writer = writer
        self.path = path
        self.codec = codec
        self.fps = fps

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


class EmotionPipeline:
    def __init__(self, emotion_model: tf.keras.Model, labels: Sequence[str], alpha: float = EMA_ALPHA_DEFAULT):
        self.model = emotion_model
        self.labels = list(labels)
        self.alpha = alpha
        self.smooth_probs: Optional[np.ndarray] = None

        self.input_hw: Tuple[int, int]
        self.input_channels: int
        self.channel_last: bool
        self._parse_input_shape()

    def _parse_input_shape(self) -> None:
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) != 4:
            raise ValueError(f"Beklenen 4D input shape, bulundu: {shape}")

        _, d1, d2, d3 = shape
        if d3 in (1, 3):
            self.channel_last = True
            self.input_hw = (int(d1), int(d2))
            self.input_channels = int(d3)
        elif d1 in (1, 3):
            self.channel_last = False
            self.input_channels = int(d1)
            self.input_hw = (int(d2), int(d3))
        else:
            raise ValueError(f"Model input kanal yapÄ±sÄ± Ã§Ã¶zÃ¼lemedi: {shape}")

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        target_w, target_h = self.input_hw[1], self.input_hw[0]

        if self.input_channels == 1:
            face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_AREA)
            face = face.astype(np.float32) / 255.0
            if self.channel_last:
                face = np.expand_dims(face, axis=-1)
            else:
                face = np.expand_dims(face, axis=0)
        elif self.input_channels == 3:
            face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_AREA)
            face = face.astype(np.float32) / 255.0
            if not self.channel_last:
                face = np.transpose(face, (2, 0, 1))
        else:
            raise ValueError(f"Desteklenmeyen kanal sayÄ±sÄ±: {self.input_channels}")

        return np.expand_dims(face, axis=0)

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        x = self.preprocess_face(face_bgr)
        probs = self.model.predict(x, verbose=0)[0].astype(np.float32)

        if self.smooth_probs is None:
            self.smooth_probs = probs
        else:
            self.smooth_probs = self.alpha * probs + (1.0 - self.alpha) * self.smooth_probs

        return self.smooth_probs


def ensure_paths() -> dict:
    paths = {
        "logs": Path("logs"),
        "screenshots": Path("logs/screenshots"),
        "videos": Path("logs/videos"),
        "sessions": Path("logs/sessions"),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_models() -> Tuple[cv2.dnn_Net, tf.keras.Model]:
    missing = [p for p in [PROTO_PATH, FACE_MODEL_PATH, EMO_MODEL_PATH] if not os.path.exists(p)]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Model dosyalarÄ± eksik:\n{missing_text}")

    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, FACE_MODEL_PATH)
    emotion_model = tf.keras.models.load_model(EMO_MODEL_PATH)
    return face_net, emotion_model


def detect_faces(frame: np.ndarray, face_net: cv2.dnn_Net, threshold: float = FACE_CONF_THRESHOLD) -> List[Tuple[float, int, int, int, int]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < threshold:
            continue

        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        faces.append((conf, x1, y1, x2, y2))

    return faces


def pick_primary_face(faces: Sequence[Tuple[float, int, int, int, int]]) -> Optional[Tuple[float, int, int, int, int]]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f[3] - f[1]) * (f[4] - f[2]))


def draw_progress_bar(panel: np.ndarray, x: int, y: int, w: int, h: int, value: float, color: Tuple[int, int, int]) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (58, 67, 80), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (80, 90, 106), 1)
    fill_w = int(max(0.0, min(1.0, value)) * (w - 4))
    if fill_w > 0:
        cv2.rectangle(panel, (x + 2, y + 2), (x + 2 + fill_w, y + h - 2), color, -1)


def draw_card(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), CARD_BG, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (47, 56, 69), 1)


def fill_vertical_gradient(img: np.ndarray, x: int, y: int, w: int, h: int, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> None:
    h_safe = max(1, h - 1)
    for i in range(h):
        t = i / h_safe
        b = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        r = int(top[2] * (1 - t) + bottom[2] * t)
        cv2.line(img, (x, y + i), (x + w, y + i), (b, g, r), 1)


def draw_chip(img: np.ndarray, x: int, y: int, text: str, active: bool) -> None:
    bg = (37, 84, 53) if active else (48, 55, 66)
    fg = (144, 255, 170) if active else (170, 178, 194)
    w = 12 + len(text) * 9
    h = 28
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 90, 106), 1)
    cv2.putText(img, text, (x + 8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg, 1, cv2.LINE_AA)


def draw_timeline(panel: np.ndarray, history: Deque[Tuple[float, int]], labels: Sequence[str], x: int, y: int, w: int, h: int) -> None:
    draw_card(panel, x, y, w, h)

    row_h = h / len(labels)
    for i in range(len(labels)):
        yy = int(y + i * row_h)
        cv2.line(panel, (x, yy), (x + w, yy), (52, 60, 72), 1)

    if len(history) < 2:
        cv2.putText(panel, "Timeline: waiting for data", (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
        return

    entries = list(history)
    t0 = entries[0][0]
    t1 = entries[-1][0]
    span = max(1e-6, t1 - t0)
    for t, idx in entries:
        xr = (t - t0) / span
        xx = int(x + xr * (w - 1))
        y1 = int(y + idx * row_h)
        y2 = int(y + (idx + 1) * row_h) - 1
        cv2.line(panel, (xx, y1), (xx, y2), ACCENT, 1)

    for i, label in enumerate(labels):
        ly = int(y + (i + 0.7) * row_h)
        cv2.putText(panel, label[:8], (x + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.37, TEXT_MUTED, 1, cv2.LINE_AA)


def compose_ui(
    frame: np.ndarray,
    labels: Sequence[str],
    probs: Optional[np.ndarray],
    dominant_idx: Optional[int],
    top3: Sequence[Tuple[str, float]],
    fps: float,
    face_count: int,
    is_logging: bool,
    is_recording: bool,
    timeline_history: Deque[Tuple[float, int]],
) -> np.ndarray:
    h, w = frame.shape[:2]

    canvas = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    overlay = canvas[:, :w].copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    canvas[:, :w] = cv2.addWeighted(overlay, 0.45, canvas[:, :w], 0.55, 0)
    cv2.putText(canvas[:, :w], "S Shot   R Record   L Log   Q Quit", (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (236, 236, 236), 2, cv2.LINE_AA)

    panel = canvas[:, w:]
    fill_vertical_gradient(panel, 0, 0, PANEL_WIDTH, h, (13, 16, 23), (22, 30, 41))
    cv2.line(panel, (0, 0), (0, h), (55, 70, 90), 1)

    cv2.putText(panel, "Emotion Dashboard", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, TEXT_MAIN, 2, cv2.LINE_AA)
    cv2.putText(panel, "Live mood stream", (22, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.line(panel, (20, 66), (PANEL_WIDTH - 20, 66), (56, 74, 96), 1)

    probs_to_draw = probs if probs is not None else np.zeros(len(labels), dtype=np.float32)
    dom_text = labels[dominant_idx] if dominant_idx is not None else "No face"
    dom_score = float(probs_to_draw[dominant_idx]) if dominant_idx is not None else 0.0

    draw_card(panel, 16, 78, PANEL_WIDTH - 32, 112)
    cv2.putText(panel, "Dominant Emotion", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, dom_text, (28, 139), cv2.FONT_HERSHEY_SIMPLEX, 1.00, ACCENT if dominant_idx is not None else TEXT_MUTED, 2, cv2.LINE_AA)
    cv2.putText(panel, f"{dom_score*100:5.1f}%", (PANEL_WIDTH - 112, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.74, TEXT_MAIN, 2, cv2.LINE_AA)
    draw_progress_bar(panel, 28, 156, PANEL_WIDTH - 56, 14, dom_score, ACCENT)

    draw_card(panel, 16, 200, PANEL_WIDTH - 32, 124)
    cv2.putText(panel, "Top 3", (28, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)
    if top3:
        row_y = 246
        for top_label, score in top3[:3]:
            cv2.putText(panel, top_label, (28, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_MAIN, 1, cv2.LINE_AA)
            draw_progress_bar(panel, 125, row_y - 10, PANEL_WIDTH - 250, 10, float(score), ACCENT)
            cv2.putText(panel, f"{score*100:4.1f}%", (PANEL_WIDTH - 104, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MAIN, 1, cv2.LINE_AA)
            row_y += 30
    else:
        cv2.putText(panel, "Face not detected", (28, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)

    prob_y = 336
    draw_card(panel, 16, prob_y, PANEL_WIDTH - 32, 214)
    cv2.putText(panel, "Emotion Probabilities", (28, prob_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)
    y_cursor = prob_y + 46
    for i, label in enumerate(labels):
        cv2.putText(panel, label, (28, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT_MAIN, 1, cv2.LINE_AA)
        draw_progress_bar(panel, 140, y_cursor - 10, 178, 10, float(probs_to_draw[i]), ACCENT)
        cv2.putText(panel, f"{probs_to_draw[i]*100:4.1f}%", (329, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MAIN, 1, cv2.LINE_AA)
        y_cursor += 26

    timeline_h = 126
    timeline_y = h - timeline_h - 14
    status_h = 56
    status_y = timeline_y - status_h - 10

    draw_card(panel, 16, status_y, PANEL_WIDTH - 32, status_h)
    cv2.putText(panel, f"FPS {fps:4.1f}", (28, status_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Faces {face_count}", (125, status_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MAIN, 1, cv2.LINE_AA)
    draw_chip(panel, 218, status_y + 14, f"REC {'ON' if is_recording else 'OFF'}", is_recording)
    draw_chip(panel, 316, status_y + 14, f"LOG {'ON' if is_logging else 'OFF'}", is_logging)

    cv2.putText(panel, "Timeline (~30s)", (20, timeline_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_MUTED, 1, cv2.LINE_AA)
    draw_timeline(panel, timeline_history, labels, 16, timeline_y, PANEL_WIDTH - 32, timeline_h)

    return canvas


def create_video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> VideoRecorder:
    candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    for codec, ext in candidates:
        target = path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(target), fourcc, fps, frame_size)
        if writer.isOpened():
            return VideoRecorder(writer, target, codec, fps)
        writer.release()

    raise RuntimeError("VideoWriter baÅŸlatÄ±lamadÄ±. mp4v/avc1/XVID/MJPG codec denemeleri baÅŸarÄ±sÄ±z.")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_camera():
    env_idx = os.getenv("CAM_INDEX")
    tried = []

    if env_idx is not None:
        try:
            idx = int(env_idx)
        except ValueError:
            raise RuntimeError(f"GeÃ§ersiz CAM_INDEX deÄŸeri: {env_idx}")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()
        raise RuntimeError(f"CAM_INDEX={idx} ile webcam aÃ§Ä±lamadÄ±.")

    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        tried.append(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()

    raise RuntimeError(f"Webcam aÃ§Ä±lamadÄ±. Denenen indexler: {tried}")


def main() -> None:
    print("[INFO] Modeller yÃ¼kleniyor...")
    try:
        face_net, emotion_model = load_models()
    except Exception as exc:
        print(f"[ERROR] Model yÃ¼kleme hatasÄ±: {exc}")
        sys.exit(1)

    try:
        pipeline = EmotionPipeline(emotion_model, EMOTIONS, alpha=EMA_ALPHA_DEFAULT)
    except Exception as exc:
        print(f"[ERROR] Duygu pipeline baÅŸlatÄ±lamadÄ±: {exc}")
        sys.exit(1)

    paths = ensure_paths()

    try:
        cap, cam_idx = open_camera()
        print(f"[INFO] Kamera aÃ§Ä±ldÄ± (index={cam_idx})")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("[ERROR] Ä°pucu: PowerShell'de geÃ§ici olarak `$env:CAM_INDEX='1'` yazÄ±p tekrar deneyin.")
        sys.exit(1)

    window_name = "Emotion Face Recognition UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_counter = 0
    fps = 0.0
    fps_timer = time.time()

    timeline_history: Deque[Tuple[float, int]] = deque(maxlen=TIMELINE_SAMPLES)

    last_probs: Optional[np.ndarray] = None
    last_dom_idx: Optional[int] = None
    last_top3: List[Tuple[str, float]] = []
    last_face_box: Optional[Tuple[int, int, int, int]] = None
    last_face_count = 0

    logger: Optional[SessionLogger] = None
    recorder: Optional[VideoRecorder] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Kameradan kare okunamadÄ±, Ã§Ä±kÄ±lÄ±yor.")
                break

            now = time.time()
            faces = detect_faces(frame, face_net)
            last_face_count = len(faces)
            best = pick_primary_face(faces)

            if best is not None:
                _, x1, y1, x2, y2 = best
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    try:
                        probs = pipeline.predict(face_roi)
                        last_probs = probs
                        last_dom_idx = int(np.argmax(probs))
                        top_idx = probs.argsort()[-3:][::-1]
                        last_top3 = [(EMOTIONS[i], float(probs[i])) for i in top_idx]
                        last_face_box = (x1, y1, x2, y2)
                        timeline_history.append((now, last_dom_idx))
                    except Exception as exc:
                        print(f"[WARN] Duygu tahmini baÅŸarÄ±sÄ±z: {exc}")

            while timeline_history and now - timeline_history[0][0] > TIMELINE_SECONDS:
                timeline_history.popleft()

            if last_face_box is not None and last_probs is not None and last_dom_idx is not None:
                x1, y1, x2, y2 = last_face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 255), 2)
                label = EMOTIONS[last_dom_idx]
                score = float(last_probs[last_dom_idx]) * 100.0
                cv2.putText(frame, f"{label} {score:.1f}%", (x1, max(18, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 220, 255), 2, cv2.LINE_AA)

            frame_counter += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_timer = now

            ui_frame = compose_ui(
                frame=frame,
                labels=EMOTIONS,
                probs=last_probs,
                dominant_idx=last_dom_idx,
                top3=last_top3,
                fps=fps,
                face_count=last_face_count,
                is_logging=logger is not None,
                is_recording=recorder is not None,
                timeline_history=timeline_history,
            )

            if logger is not None and last_probs is not None and last_dom_idx is not None:
                logger.log(datetime.now().isoformat(timespec="seconds"), EMOTIONS[last_dom_idx], last_probs)

            if recorder is not None:
                recorder.write(ui_frame)

            cv2.imshow(window_name, ui_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                shot_path = paths["screenshots"] / f"shot_{timestamp_now()}.png"
                if cv2.imwrite(str(shot_path), ui_frame):
                    print(f"[INFO] Screenshot kaydedildi: {shot_path}")
                else:
                    print("[WARN] Screenshot kaydedilemedi.")
            elif key in (ord("l"), ord("L")):
                if logger is None:
                    session_path = paths["sessions"] / f"session_{timestamp_now()}.csv"
                    logger = SessionLogger(session_path, EMOTIONS)
                    print(f"[INFO] Log baÅŸladÄ±: {session_path}")
                else:
                    logger.close()
                    print(f"[INFO] Log kapandÄ±: {logger.path}")
                    logger = None
            elif key in (ord("r"), ord("R")):
                if recorder is None:
                    base = paths["videos"] / f"session_{timestamp_now()}"
                    try:
                        recorder = create_video_writer(base, max(1.0, fps), (ui_frame.shape[1], ui_frame.shape[0]))
                        print(f"[INFO] KayÄ±t baÅŸladÄ±: {recorder.path} (codec={recorder.codec})")
                    except Exception as exc:
                        print(f"[WARN] Video kaydÄ± baÅŸlatÄ±lamadÄ±: {exc}")
                else:
                    recorder.close()
                    print(f"[INFO] KayÄ±t kapandÄ±: {recorder.path}")
                    recorder = None

    finally:
        if logger is not None:
            logger.close()
        if recorder is not None:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

