import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DATA_DIR = os.path.join("data", "rafdb")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

IMG_SIZE = 96
BATCH = 32
EPOCHS = 15

# Senin proje sırası (0..6)
# RAF-DB klasörleri 1..7:
# 1:surprise 2:fear 3:disgust 4:happiness 5:sadness 6:anger 7:neutral
RAF_ID_TO_TR = {
    1: "Şaşkın",
    2: "Korku",
    3: "İğrenme",
    4: "Mutlu",
    5: "Üzgün",
    6: "Kızgın",
    7: "Nötr",
}
EMOTIONS_TR = ["Kızgın","İğrenme","Korku","Mutlu","Üzgün","Şaşkın","Nötr"]
TR_TO_IDX = {name: i for i, name in enumerate(EMOTIONS_TR)}

def load_split(split_dir):
    images = []
    labels = []

    for class_id in range(1, 8):
        class_folder = os.path.join(split_dir, str(class_id))
        if not os.path.isdir(class_folder):
            print(f"[WARN] klasör yok: {class_folder}")
            continue

        tr_name = RAF_ID_TO_TR[class_id]
        y_idx = TR_TO_IDX[tr_name]

        # klasördeki tüm jpg/png dosyaları
        for fname in os.listdir(class_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(class_folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(y_idx)

    if len(images) == 0:
        raise RuntimeError(f"Hiç görüntü okunamadı: {split_dir}")

    x = np.stack(images, axis=0).astype(np.float32)  # (N,96,96,3)
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=7)  # (N,7)
    return x, y

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
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(7, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading train...")
    x_train, y_train = load_split(TRAIN_DIR)
    print("Loading test...")
    x_test, y_test = load_split(TEST_DIR)

    print("Train:", x_train.shape, y_train.shape)
    print("Test :", x_test.shape, y_test.shape)

    # Augmentation (hafif)
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

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(3000).batch(BATCH)
    train_ds = train_ds.map(lambda a,b: (aug(a, training=True), b)).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
    print("Saved -> models/emotion_rafdb.h5")

if __name__ == "__main__":
    main()
