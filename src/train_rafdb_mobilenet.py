import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# =========================
# AYARLAR
# =========================
RAF_DIR = os.path.join("data", "rafdb")  # <-- RAF-DB klasör adın neyse bunu ona göre değiştir
LABEL_FILE = os.path.join(RAF_DIR, "list_patition_label.txt")

# RAF-DB etiket mapping (RAF: 1..7)
# 1:surprise 2:fear 3:disgust 4:happiness 5:sadness 6:anger 7:neutral
RAF_TO_TR = {
    1: "Şaşkın",
    2: "Korku",
    3: "İğrenme",
    4: "Mutlu",
    5: "Üzgün",
    6: "Kızgın",
    7: "Nötr",
}

# Senin proje sırası (FER2013 ile aynı tutuyoruz)
EMOTIONS_TR = ["Kızgın", "İğrenme", "Korku", "Mutlu", "Üzgün", "Şaşkın", "Nötr"]
TR_TO_IDX = {name: i for i, name in enumerate(EMOTIONS_TR)}

IMG_SIZE = 96
BATCH = 32
EPOCHS = 15
LR = 1e-4

def find_image_path(fname: str) -> str:
    """
    RAF-DB farklı paketlerde farklı klasör yapısıyla gelebiliyor.
    En yaygın 2-3 yolu dener.
    """
    candidates = [
        os.path.join(RAF_DIR, "Image", "aligned", fname),
        os.path.join(RAF_DIR, "aligned", fname),
        os.path.join(RAF_DIR, "images", fname),
        os.path.join(RAF_DIR, fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

def load_rafdb():
    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(
            f"Label dosyası bulunamadı: {LABEL_FILE}\n"
            f"RAF_DIR'i doğru ayarla ve RAF-DB içindeki list_patition_label.txt dosyasını kontrol et."
        )

    x_train, y_train, x_test, y_test = [], [], [], []

    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # örn: train_00001.jpg 6
            parts = line.split()
            if len(parts) != 2:
                continue
            fname, lab = parts[0], int(parts[1])
            tr_name = RAF_TO_TR.get(lab, None)
            if tr_name is None:
                continue
            y_idx = TR_TO_IDX[tr_name]

            img_path = find_image_path(fname)
            if not img_path:
                # bazı paketlerde isimler uzantısız olabiliyor, jpg dene
                if not fname.lower().endswith(".jpg"):
                    img_path = find_image_path(fname + ".jpg")
            if not img_path:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0

            if fname.startswith("train"):
                x_train.append(img); y_train.append(y_idx)
            else:
                x_test.append(img); y_test.append(y_idx)

    x_train = np.array(x_train, dtype=np.float32)
    y_train = tf.keras.utils.to_categorical(np.array(y_train), num_classes=7)
    x_test  = np.array(x_test, dtype=np.float32)
    y_test  = tf.keras.utils.to_categorical(np.array(y_test), num_classes=7)

    print("Train:", x_train.shape, y_train.shape)
    print("Test :", x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(7, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    x_train, y_train, x_test, y_test = load_rafdb()

    # Basit augmentation
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/emotion_rafdb.h5", save_best_only=True),
    ]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(2000).batch(BATCH)
    train_ds = train_ds.map(lambda a,b: (aug(a, training=True), b)).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
    print("Saved -> models/emotion_rafdb.h5")

if __name__ == "__main__":
    main()