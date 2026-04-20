import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

DATA_DIR = "data/rafdb"

IMG_SIZE = 96
BATCH = 32
EPOCHS = 15

EMOTIONS_TR = ["Kızgın","İğrenme","Korku","Mutlu","Üzgün","Şaşkın","Nötr"]

def load_data():
    train_csv = os.path.join(DATA_DIR, "train_labels.csv")
    test_csv  = os.path.join(DATA_DIR, "test_labels.csv")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # CSV kolonlarını otomatik bul (bazı datasetlerde farklı isim)
    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    img_col_train = pick_col(train_df, ["image", "Image", "image_name", "Image name", "imagename", "img"])
    lab_col_train = pick_col(train_df, ["label", "Label", "class", "Class"])

    img_col_test = pick_col(test_df, ["image", "Image", "image_name", "Image name", "imagename", "img"])
    lab_col_test = pick_col(test_df, ["label", "Label", "class", "Class"])

    if img_col_train is None or lab_col_train is None:
        raise ValueError(f"train_labels.csv kolonları beklenmedik. Kolonlar: {list(train_df.columns)}")
    if img_col_test is None or lab_col_test is None:
        raise ValueError(f"test_labels.csv kolonları beklenmedik. Kolonlar: {list(test_df.columns)}")

    def process(df, folder, img_col, lab_col):
        images = []
        labels = []

        for _, row in df.iterrows():
            fname = str(row[img_col])
            # bazen uzantısız gelir
            if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg")):
                fname = fname + ".jpg"

            img_path = os.path.join(DATA_DIR, folder, fname)
            img = cv2.imread(img_path)

            if img is None:
                # dosya bulunamadıysa geç (ama sayalım)
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0  # (96,96,3)

            images.append(img)

            # label bazen 1..7 bazen 0..6 olabilir
            lab = int(row[lab_col])
            if lab >= 1:
                lab = lab - 1
            labels.append(lab)

        if len(images) == 0:
            raise RuntimeError(
                f"{folder} klasöründen hiç görüntü okunamadı.\n"
                f"Kontrol: {DATA_DIR}/{folder} içinde resimler var mı?\n"
                f"Örnek yol: {os.path.join(DATA_DIR, folder)}"
            )

        x = np.stack(images, axis=0).astype(np.float32)          # (N,96,96,3)
        y = tf.keras.utils.to_categorical(np.array(labels), 7)   # (N,7)
        return x, y

    x_train, y_train = process(train_df, "train", img_col_train, lab_col_train)
    x_test, y_test   = process(test_df, "test",  img_col_test,  lab_col_test)

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
    x = base(inp, training=False)
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
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "models/emotion_rafdb.h5",
            save_best_only=True
        )
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
