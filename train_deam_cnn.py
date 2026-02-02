import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf

# === CONFIG ===
DEAM_ROOT = os.path.join(os.path.dirname(__file__), "DEAM")

ANNOT_PATH = os.path.join(
    DEAM_ROOT,
    "DEAM_Annotations",
    "annotations",
    "annotations averaged per song",
    "song_level",
    "static_annotations_averaged_songs_1_2000.csv",  # change if your file name is different
)

AUDIO_DIR = os.path.join(DEAM_ROOT, "DEAM_audio")

# NOTE: your CSV has leading spaces in the column names for valence/arousal
ID_COL = "song_id"
VAL_COL = " valence_mean"
ARO_COL = " arousal_mean"

SR = 22050
N_MELS = 128
FRAMES = 128
MAX_SONGS = 300  # limit for speed so it runs in reasonable time


# === HELPERS ===

def list_audio_files():
    files = glob.glob(os.path.join(AUDIO_DIR, "**", "*.mp3"), recursive=True)
    files = sorted(files)
    print("Total audio files:", len(files))
    print("First few audio files:", [os.path.basename(f) for f in files[:10]])
    return files


def load_melspectrogram(path):
    """Load one track and convert to fixed-size mel-spectrogram."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    # Take first 30 seconds max
    max_samples = SR * 30
    if len(y) > max_samples:
        y = y[:max_samples]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise to roughly [0,1]
    mel_db = (mel_db + 80.0) / 80.0

    # Fix time dimension to FRAMES
    if mel_db.shape[1] < FRAMES:
        pad = FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_db = mel_db[:, :FRAMES]

    # Shape (H, W, 1) for CNN
    return mel_db[..., np.newaxis]


# === MODEL ===

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="tanh"),  # valence, arousal in [-1,1]
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# === MAIN ===

def main():
    # 1) Load annotations
    ann_path = ANNOT_PATH
    df = pd.read_csv(ann_path)

    print("Columns in annotation CSV:")
    print(df.columns.tolist())
    print("Example song_ids from CSV:", df[ID_COL].head().tolist())

    if ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' not found. Check the CSV and update ID_COL.")
    if VAL_COL not in df.columns or ARO_COL not in df.columns:
        raise ValueError("Valence/arousal columns not found. Check CSV and update VAL_COL/ARO_COL.")

    df = df[[ID_COL, VAL_COL, ARO_COL]].dropna()

    # 2) List audio files
    audio_files = list_audio_files()

    # 3) Sort both and pair by index (simple but works for project)
    df = df.sort_values(ID_COL).reset_index(drop=True)
    N = min(MAX_SONGS, len(df), len(audio_files))
    print("Using N =", N)

    X_list = []
    y_list = []

    for i in range(N):
        row = df.iloc[i]
        path = audio_files[i]
        try:
            mel = load_melspectrogram(path)
        except Exception as e:
            print("Skip", path, "error:", e)
            continue

        X_list.append(mel)
        y_list.append([row[VAL_COL], row[ARO_COL]])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    print("Data shape:", X.shape, y.shape)

    # 4) Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Build and train model
    model = build_model(X_train.shape[1:])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=8
    )

    model.save("deam_cnn_valence_arousal.h5")
    print("Model saved to deam_cnn_valence_arousal.h5")


if __name__ == "__main__":
    main()
