import os
import urllib.request
import cv2

MODELS_DIR = "models"

# OpenCV'nin örnek prototxt dosyası (genelde stabil)
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# ✅ 404 vermeyen güncel model yolu (fp16)
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

PROTO_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")


def safe_download(url: str, path: str, desc: str):
    """
    Dosyayı indirir. Hatalı/yarım indirme olursa dosyayı silip tekrar dener.
    """
    # Eğer dosya var ama 0 byte ise bozuk sayalım
    if os.path.exists(path) and os.path.getsize(path) == 0:
        try:
            os.remove(path)
        except OSError:
            pass

    if os.path.exists(path):
        return

    print(f"Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        # Bozuk dosya oluştuysa temizle
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise RuntimeError(f"{desc} indirilemedi. Hata: {e}")

    # Son kontrol: dosya gerçekten indi mi?
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise RuntimeError(f"{desc} indirildi ama dosya bozuk görünüyor (çok küçük). İnternet kopmuş olabilir.")


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_download(PROTO_URL, PROTO_PATH, "deploy.prototxt")
    safe_download(MODEL_URL, MODEL_PATH, "face detector caffemodel")


def main():
    ensure_models()

    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam açılamadı. Kamera iznini kontrol et (0 yerine 1 de deneyebilirsin).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.60:
                continue

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Test - OpenCV DNN (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()