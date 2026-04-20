import os
import cv2
import numpy as np
import tensorflow as tf

# ==============================
# MODEL YOLLARI
# ==============================
MODELS_DIR = "models"
PROTO_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
EMO_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn.h5")

# FER2013 sırası
EMOTIONS = ["Kizgin", "İğrenme", "Korku", "Mutlu", "Uzgun", "Şaşkın", "Notr"]


def preprocess_face(gray_face_48):
    """(48,48) grayscale -> (1,48,48,1) float32 normalized"""
    x = gray_face_48.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x


def draw_header_panel(frame, main_label, main_score, top_items):
    """
    Üstte şık panel çiz.
    top_items: [(label, prob), ...]  # top-3
    """
    h, w = frame.shape[:2]
    panel_h = 72

    # yarı saydam panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # ana satır
    title = f"Duygu: {main_label}  ({main_score*100:.1f}%)"
    cv2.putText(frame, title, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    # top-3 satırı
    items = " | ".join([f"{lbl}:{p*100:.0f}%" for (lbl, p) in top_items])
    cv2.putText(frame, items, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    # dosya kontrol
    if not os.path.exists(PROTO_PATH) or not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError("Face detector dosyaları eksik (models/ içinde deploy.prototxt ve .caffemodel olmalı).")

    if not os.path.exists(EMO_MODEL_PATH):
        raise FileNotFoundError(f"Emotion modeli bulunamadı: {EMO_MODEL_PATH}")

    print("Modeller yükleniyor...")
    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, FACE_MODEL_PATH)
    emo_model = tf.keras.models.load_model(EMO_MODEL_PATH)

    print("Kamera açılıyor...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam açılamadı. (0 yerine 1 deneyebilirsin)")

    # tahmin yumuşatma (EMA)
    smooth_probs = None
    alpha = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ==========================
        # FACE DETECT
        # ==========================
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        face_net.setInput(blob)
        detections = face_net.forward()

        best = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if best is None or conf > best[0]:
                best = (conf, x1, y1, x2, y2)

        # ==========================
        # EMOTION
        # ==========================
        if best is not None:
            conf, x1, y1, x2, y2 = best

            # yüz kutusu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

                x = preprocess_face(gray48)
                probs = emo_model.predict(x, verbose=0)[0]

                # EMA stabilizasyon
                if smooth_probs is None:
                    smooth_probs = probs
                else:
                    smooth_probs = alpha * probs + (1 - alpha) * smooth_probs
                probs = smooth_probs

                idx = int(np.argmax(probs))
                score = float(probs[idx])
                label = EMOTIONS[idx]

                # top-3
                top3_idx = probs.argsort()[-3:][::-1]
                top3_items = [(EMOTIONS[j], float(probs[j])) for j in top3_idx]

                # ÜST PANELDE GÖSTER
                draw_header_panel(frame, label, score, top3_items)

        cv2.imshow("Canli Duygu Analizi (q cikis)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()