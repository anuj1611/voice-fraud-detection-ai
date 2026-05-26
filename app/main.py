from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .audio_utils import preprocess_audio
from .model_loader import predict
from .config import API_KEY

app = FastAPI(title="AI Voice Detection API")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.get("/")
def ui_home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    try:
        features = preprocess_audio(request.audioBase64)
        score = predict(features)

        classification = "AI_GENERATED" if score > 0.5 else "HUMAN"

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(score, 4),
            "explanation": "Classification based on voice spectral patterns and prosody analysis"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
