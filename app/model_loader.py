import tensorflow as tf
from .config import MODEL_PATH

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict(features):
    prediction = model.predict(features)[0][0]
    return float(prediction)
