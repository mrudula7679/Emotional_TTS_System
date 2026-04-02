import os
import librosa, soundfile as sf
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_WAV_DIR = "data/processed/wavs"
os.makedirs(OUT_WAV_DIR, exist_ok=True)
EMOTION_MAP = {"neutral":0,"happy":1,"sad":2,"angry":3,"fear":4,"disgust":5,"surprise":6}

def save_wav_and_meta(src_path, dataset_tag, speaker_id, emotion_label, out_dir=OUT_WAV_DIR, sr=22050):
    fname = f"{dataset_tag}_{os.path.basename(src_path)}"
    out_path = os.path.join(out_dir, fname)
    y, _ = librosa.load(src_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
    sf.write(out_path, y, sr)
    return out_path, speaker_id, EMOTION_MAP[emotion_label]

def collect_all():
    rows = []
    # RAVDESS
    ravdess_root = os.path.join(RAW_DIR, "ravdess")
    # assume Actor_* dirs -> map Actor_01 -> speaker id (0-based)
    actor_dirs = sorted([d for d in os.listdir(ravdess_root) if d.startswith("Actor")])
    actor2id = {actor: idx for idx, actor in enumerate(actor_dirs)}
    for actor in actor_dirs:
        aroot = os.path.join(ravdess_root, actor)
        for f in os.listdir(aroot):
            if not f.endswith(".wav"): continue
            emotion_id = int(f.split("-")[2])
            emotion_map = {1:"neutral",3:"happy",4:"sad",5:"angry",6:"fear",7:"disgust",8:"surprise"}
            emotion_label = emotion_map.get(emotion_id, "neutral")
            wav_path = os.path.join(aroot, f)
            out_wav, spk, emo = save_wav_and_meta(wav_path, "ravdess", actor2id[actor], emotion_label)
            rows.append((out_wav, " ".join([]) , spk, emo))  # transcript placeholder if you lack transcript

    # CREMA-D (AudioWAV/1001_DFA_ANG_...)
    crema_root = os.path.join(RAW_DIR, "crema-d", "AudioWAV")
    # create speaker map from file prefixes (first 4 digits)
    speakers = sorted(list({f.split("_")[0] for f in os.listdir(crema_root) if f.endswith(".wav")}))
    spk2id = {s:i for i,s in enumerate(speakers)}
    for f in os.listdir(crema_root):
        if not f.endswith(".wav"): continue
        parts = f.split("_")
        spk = parts[0]
        emo_code = parts[2]
        emo_map = {"ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral","SAD":"sad"}
        emo_label = emo_map.get(emo_code, "neutral")
        wav_path = os.path.join(crema_root, f)
        out_wav, spk_id, emo = save_wav_and_meta(wav_path, "cremad", spk2id[spk], emo_label)
        rows.append((out_wav, " ", spk_id, emo))

    # TESS (folders like OAF_angry)
    tess_root = os.path.join(RAW_DIR, "tess")
    # each file name pattern: *_<emotion>_*.wav or tree by folder
    tess_speakers = {}
    spk_counter = 0
    for root, _, files in os.walk(tess_root):
        for f in files:
            if not f.endswith(".wav"): continue
            # try to infer emotion from folder or filename
            folder = os.path.basename(root)
            # folder often like OAF_angry -> split by '_' last token
            if "_" in folder:
                emo = folder.split("_")[-1].lower()
            else:
                emo = f.split("_")[1].lower() if "_" in f else "neutral"
            emo_map = {"angry":"angry","disgust":"disgust","fear":"fear","happy":"happy","neutral":"neutral","ps":"surprise","sad":"sad"}
            emo_label = emo_map.get(emo, "neutral")
            # create a speaker id per folder
            spk = folder
            if spk not in tess_speakers:
                tess_speakers[spk] = spk_counter; spk_counter += 1
            wav_path = os.path.join(root, f)
            out_wav, spk_id, emo = save_wav_and_meta(wav_path, "tess", tess_speakers[spk], emo_label)
            rows.append((out_wav, " ".join([]), spk_id, emo))

    # Now split and write metadata
    import random
    random.shuffle(rows)
    split_idx = int(0.9 * len(rows))
    train = rows[:split_idx]
    val = rows[split_idx:]

    def write_meta(list_rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for wav, text, spk, emo in list_rows:
                fh.write(f"{wav}|{text}|{spk}|{emo}\n")

    os.makedirs("data/processed", exist_ok=True)
    write_meta(train, "data/processed/metadata_train.txt")
    write_meta(val, "data/processed/metadata_val.txt")
    print("Wrote", len(train), "train and", len(val), "val samples.")

if __name__ == "__main__":
    collect_all()
