import librosa
import numpy as np
import base64
import os
import tempfile
from .config import SAMPLE_RATE, N_MELS, MAX_PAD_LEN


def preprocess_audio(base64_audio: str):
    audio_bytes = base64.b64decode(base64_audio)

    
    temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.mp3")

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_PAD_LEN]

    mel_db = np.expand_dims(mel_db, axis=(0, -1))
    return mel_db
