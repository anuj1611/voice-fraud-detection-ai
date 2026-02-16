import os
import librosa
import numpy as np
from tqdm import tqdm

DATASET_PATH = "training/dataset"
OUTPUT_X = []
OUTPUT_Y = []

SAMPLE_RATE = 16000
N_MELS = 128
MAX_PAD_LEN = 128  


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        
        if mel_db.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_PAD_LEN]

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


for label, category in enumerate(["human", "ai"]):
    folder = os.path.join(DATASET_PATH, category)
    print(f"\nProcessing {category} voices...")

    for file in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file)

        features = extract_features(file_path)
        if features is not None:
            OUTPUT_X.append(features)
            OUTPUT_Y.append(label)  

X = np.array(OUTPUT_X)
y = np.array(OUTPUT_Y)


X = X[..., np.newaxis]

print("\nFinal Feature Shape:", X.shape)
print("Labels Shape:", y.shape)


np.save("training/X_features.npy", X)
np.save("training/y_labels.npy", y)

print("\n Feature extraction complete!")
