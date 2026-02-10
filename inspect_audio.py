import os, torchaudio
from pathlib import Path

BASE = Path(__file__).parent / "data" / "audio_organized"

for split in ["train", "val"]:
    for cls in ["cats", "dogs"]:
        d = str(BASE / split / cls)
        files = [f for f in os.listdir(d) if f.endswith(".wav")][:2]
        for f in files:
            path = os.path.join(d, f)
            waveform, sr = torchaudio.load(path)
            dur = waveform.shape[1] / sr
            print(f"[{split}/{cls}] {f}: sr={sr}, ch={waveform.shape[0]}, frames={waveform.shape[1]}, dur={dur:.2f}s")
