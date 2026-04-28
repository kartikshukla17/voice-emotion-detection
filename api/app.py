"""FastAPI server for speech emotion recognition (Wav2Vec2 backbone).

Run:
    uvicorn api.app:app --reload --port 8000

POST /predict accepts an audio file (.wav, .mp3, .ogg, .flac, .m4a, .webm)
and returns the predicted emotion + per-class probabilities.
"""

import io
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "emotion_model.pkl"
MIN_AUDIO_SECONDS = 0.25
ALLOWED_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm")

bundle = joblib.load(MODEL_PATH)
classifier = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
classes = [str(c) for c in label_encoder.classes_]
WAV2VEC_NAME = bundle.get("wav2vec_model", "facebook/wav2vec2-base")
TARGET_SR = bundle.get("target_sr", 16000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {WAV2VEC_NAME} on {device}...")
w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_NAME)
w2v_model = Wav2Vec2Model.from_pretrained(WAV2VEC_NAME).to(device).eval()
print("Wav2Vec2 ready.")

app = FastAPI(title="Speech Emotion Recognition API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@torch.no_grad()
def extract_embedding(audio: np.ndarray) -> np.ndarray:
    """Run Wav2Vec2 and mean-pool the last hidden state across time."""
    audio = np.asarray(audio, dtype=np.float32)
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) >= TARGET_SR // 4:
        audio = trimmed
    inputs = w2v_processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
    outputs = w2v_model(inputs.input_values.to(device))
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


@app.get("/")
def health():
    return {
        "status": "ok",
        "classifier": bundle.get("model_name", "unknown"),
        "feature_extractor": bundle.get("feature_extractor", "unknown"),
        "wav2vec_model": WAV2VEC_NAME,
        "feature_dim": int(bundle.get("feature_dim", -1)),
        "classes": classes,
        "sample_rate": TARGET_SR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    fname = (file.filename or "").lower()
    if not fname.endswith(ALLOWED_EXTS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Use one of: {ALLOWED_EXTS}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        audio, _ = librosa.load(io.BytesIO(contents), sr=TARGET_SR, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")

    if len(audio) < TARGET_SR * MIN_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short — need at least {MIN_AUDIO_SECONDS}s",
        )

    embedding = extract_embedding(audio).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)

    pred_idx = int(classifier.predict(embedding_scaled)[0])
    pred_emotion = classes[pred_idx]

    response = {
        "emotion": pred_emotion,
        "confidence": None,
        "all_scores": {},
        "duration_seconds": round(len(audio) / TARGET_SR, 3),
    }

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(embedding_scaled)[0]
        response["confidence"] = float(probs[pred_idx])
        response["all_scores"] = {classes[i]: float(probs[i]) for i in range(len(probs))}

    return response
