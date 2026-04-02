# train_audio_emotion_model.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib  # pip install joblib

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

def extract_features(path, n_mfcc=40):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # [web:54][web:64][web:67]
    return np.mean(mfcc, axis=1)

def load_dataset(root):
    X, y = [], []
    for emo in EMOTIONS:
        emo_dir = os.path.join(root, emo)
        if not os.path.isdir(emo_dir):
            continue
        for fname in os.listdir(emo_dir):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(emo_dir, fname)
            feat = extract_features(fpath)
            X.append(feat)
            y.append(emo)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_root = "data/audio_emotion_train"  # put your WAVs in subfolders: happy/, sad/, ...
    X, y = load_dataset(data_root)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # [web:35][web:44]

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Train+test accuracy (quick check):", accuracy_score(y_test, y_pred))  # [web:47][web:71]

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/audio_emotion_model.joblib")
    print("Saved model to models/audio_emotion_model.joblib")
