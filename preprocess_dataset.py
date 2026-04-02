import os
import librosa
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import random
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
WAVS_DIR = os.path.join(PROCESSED_DIR, "wavs")
os.makedirs(WAVS_DIR, exist_ok=True)

# Emotions map
EMOTION_MAP = {
    "angry": "angry", "happy": "happy", "sad": "sad", "fear": "fear",
    "disgust": "disgust", "surprise": "surprise", "neutral": "neutral", "calm": "neutral"
}

# RAVDESS: Actors 01-12 = Female, 13-24 = Male
RAVDESS_GENDER_MAP = {}
for i in range(1, 13):  # 01-12 female
    RAVDESS_GENDER_MAP[f"Actor_{i:02d}"] = "female"
for i in range(13, 25):  # 13-24 male
    RAVDESS_GENDER_MAP[f"Actor_{i:02d}"] = "male"

def process_and_save(file_path, target_path, sr=22050):
    """Load audio, resample, convert to mono, normalize, save"""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y = librosa.util.normalize(y)
        sf.write(target_path, y, sr)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def parse_ravdess_filename(filename):
    """Robust RAVDESS filename parser: Actor_01-Modal-Emotion-Repeat.wav"""
    try:
        # Expected format: Actor_01-03-01-01-0000000000.wav or Actor_01-Modal-Emotion-Repeat.wav
        parts = filename.replace('.wav', '').split('-')
        if len(parts) >= 4:
            actor_str = parts[0]  # Actor_01
            emotion_id = int(parts[2])  # Emotion code
            
            # RAVDESS emotion mapping
            emotion_codes = {
                1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                5: "angry", 6: "fear", 7: "disgust", 8: "surprise"
            }
            emotion = emotion_codes.get(emotion_id, "neutral")
            gender = RAVDESS_GENDER_MAP.get(actor_str, "female")
            
            return actor_str, emotion, gender
        return None, "neutral", "female"
    except:
        return None, "neutral", "female"

def collect_metadata():
    """Iterate datasets and create metadata list"""
    metadata = []
    processed_count = 0

    #  RAVDESS (Highest precision - both genders)
    print(" Processing RAVDESS...")
    ravdess_path = os.path.join(RAW_DIR, "ravdess")
    if os.path.exists(ravdess_path):
        for root, _, files in os.walk(ravdess_path):
            for f in files:
                if f.endswith(".wav"):
                    actor_str, emotion, gender = parse_ravdess_filename(f)
                    new_name = f"ravdess_{gender}_{f}"
                    target_path = os.path.join(WAVS_DIR, new_name)
                    
                    if process_and_save(os.path.join(root, f), target_path):
                        metadata.append([target_path, "dummy text", 
                                    EMOTION_MAP.get(emotion, "neutral"), gender])
                        processed_count += 1
    else:
        print(f"⚠️  RAVDESS path not found: {ravdess_path}")

    # 2️CREMA-D (High quality adult voices)
    print("Processing CREMA-D...")
    crema_path = os.path.join(RAW_DIR, "crema-d", "AudioWAV")
    if os.path.exists(crema_path):
        for f in os.listdir(crema_path):
            if f.endswith(".wav"):
                parts = f.split("_")
                if len(parts) >= 3:
                    actor_id = parts[0]
                    emotion_code = parts[2]
                    
                    # Gender: 4xxx = female, others = male
                    gender = "female" if actor_id.startswith("4") else "male"
                    
                    emotion_map = {
                        "ANG": "angry", "DIS": "disgust", "FEA": "fear", 
                        "HAP": "happy", "NEU": "neutral", "SAD": "sad"
                    }
                    emotion = emotion_map.get(emotion_code, "neutral")
                    
                    new_name = f"cremad_{gender}_{f}"
                    target_path = os.path.join(WAVS_DIR, new_name)
                    
                    if process_and_save(os.path.join(crema_path, f), target_path):
                        metadata.append([target_path, "dummy text", 
                                    EMOTION_MAP.get(emotion, "neutral"), gender])
                        processed_count += 1
    else:
        print(f"  CREMA-D path not found: {crema_path}")

    print(f" Processed {processed_count} adult audio files from {len(metadata)} total samples")
    return metadata

def main():
    print(" Creating highest precision adult-only dataset (Male/Female only)...")
    metadata = collect_metadata()
    
    # Filter only adult voices with proper metadata
    adult_metadata = [row for row in metadata if len(row) == 4 and row[3] in ["male", "female"]]
    
    # Gender statistics
    female_count = sum(1 for row in adult_metadata if row[3] == "female")
    male_count = sum(1 for row in adult_metadata if row[3] == "male")
    
    print(f"  Female samples: {female_count}")
    print(f"  Male samples: {male_count}")
    print(f" Total adult samples: {len(adult_metadata)}")
    
    if len(adult_metadata) == 0:
        print(" No valid adult samples found! Check your data/raw/ directory structure.")
        return
    
    # Train/Val split
    train, val = train_test_split(adult_metadata, test_size=0.1, random_state=42)
    
    train_path = os.path.join(PROCESSED_DIR, "metadata_train.txt")
    val_path = os.path.join(PROCESSED_DIR, "metadata_val.txt")
    
    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    pd.DataFrame(train).to_csv(train_path, sep="|", index=False, header=False)
    pd.DataFrame(val).to_csv(val_path, sep="|", index=False, header=False)
    
    print(f" Train metadata saved: {train_path} ({len(train)} samples)")
    print(f" Val metadata saved: {val_path} ({len(val)} samples)")
    print(" Adult-only dataset ready with proper gender separation!")

if __name__ == "__main__":
    main()
