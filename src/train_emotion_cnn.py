import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_PATH = "data/fer2013.csv"
MODEL_OUT = "models/emotion_cnn.h5"

def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)

    def parse_pixels(pix):
        arr = np.fromstring(pix, sep=" ", dtype=np.float32)
        return arr.reshape(48, 48)

    X = np.stack(df["pixels"].apply(parse_pixels).values)
    y = df["emotion"].astype(int).values

    X = X / 255.0
    X = X[..., np.newaxis]

    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"] == "PublicTest"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return (X_train, y_train), (X_val, y_val)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    os.makedirs("models", exist_ok=True)

    (X_train, y_train), (X_val, y_val) = load_fer2013(DATA_PATH)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)

    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64
    )

    model.save(MODEL_OUT)
    print("Model kaydedildi:", MODEL_OUT)


if __name__ == "__main__":
    main()

    