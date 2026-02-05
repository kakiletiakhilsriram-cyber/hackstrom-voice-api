from fastapi import FastAPI, Header, HTTPException
import base64
import os
import json
import numpy as np
import librosa
import google.generativeai as genai
import tempfile

app = FastAPI()

API_KEY = "ShieldAI206"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def root():
    return {"status": "Hackstrom Voice API running (Gemini reasoning mode)"}

@app.post("/analyze")
def analyze(data: dict, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if "audio_base64" not in data:
        raise HTTPException(status_code=400, detail="Missing audio_base64")

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(data["audio_base64"])
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Save temp mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        y, sr = librosa.load(temp_path, sr=None)
    except:
        raise HTTPException(status_code=400, detail="Audio decoding failed")

    # Feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

    feature_summary = {
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc)),
        "spectral_centroid": float(spectral_centroid),
        "zero_crossing_rate": float(zero_crossing)
    }

    prompt = f"""
You are an AI voice forensic system.

Given the following extracted speech features, determine whether the voice
is AI-generated or spoken by a human.

Supported languages:
Tamil, English, Hindi, Malayalam, Telugu.

Speech Features:
{json.dumps(feature_summary, indent=2)}

Return ONLY valid JSON in this format:
{{
  "prediction": "AI_GENERATED" or "HUMAN",
  "confidence": number between 0 and 1,
  "explanation": "short reason"
}}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except:
        raise HTTPException(status_code=500, detail="Gemini reasoning failed")
