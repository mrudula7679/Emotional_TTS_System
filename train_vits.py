import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd

# Simple dataset wrapper
class EmotionSpeechDataset(Dataset):
    def __init__(self, metadata_file, sr=22050):
        self.df = pd.read_csv(metadata_file, sep="|", header=None, names=["path", "text", "emotion"])
        self.sr = sr
        self.emotion2id = {e: i for i, e in enumerate(sorted(self.df["emotion"].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, _ = librosa.load(row["path"], sr=self.sr)
        wav = librosa.util.normalize(wav)

        return {
            "audio": torch.tensor(wav, dtype=torch.float32),
            "text": row["text"],
            "emotion": self.emotion2id[row["emotion"]]
        }

# Dummy placeholder VITS-like model
class SimpleVITS(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions)
        )

    def forward(self, mel):
        return self.fc(mel)

def train_model(train_metadata, val_metadata, epochs=10, batch_size=4, lr=1e-4):
    train_dataset = EmotionSpeechDataset(train_metadata)
    val_dataset = EmotionSpeechDataset(val_metadata)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = SimpleVITS(num_emotions=len(train_dataset.emotion2id))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            mel = torch.stft(batch["audio"], n_fft=400, return_complex=True).abs().mean(dim=-1)
            emotion = batch["emotion"]

            preds = model(mel)
            loss = criterion(preds, emotion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vits_emotion.pth")
    print(" Training complete. Model saved at checkpoints/vits_emotion.pth")

if __name__ == "__main__":
    train_model(
        "data/processed/metadata_train.txt",
        "data/processed/metadata_val.txt"
    )
