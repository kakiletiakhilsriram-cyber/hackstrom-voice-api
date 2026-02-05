from fastapi import FastAPI, Header, HTTPException
import base64
import google.generativeai as genai
import json
import os

app = FastAPI()

API_KEY = "ShieldAI206"

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def root():
    return {"status": "Hackstrom Voice API running with Gemini"}

@app.post("/analyze")
def analyze(data: dict, x_api_key: str = Header(None)):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if "audio_base64" not in data:
        raise HTTPException(status_code=400, detail="Missing audio_base64")

    try:
        audio_bytes = base64.b64decode(data["audio_base64"])
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Gemini model
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = """
You are an AI voice forensic system.

Analyze the given MP3 audio and determine whether the voice
is AI-generated or spoken by a human.

Supported languages:
Tamil, English, Hindi, Malayalam, Telugu.

Focus on:
- Pitch variability
- Prosody naturalness
- Spectral smoothness
- Timing irregularities common in TTS systems

Return ONLY valid JSON in this format:
{
  "prediction": "AI_GENERATED" or "HUMAN",
  "confidence": number between 0 and 1
}
"""

    try:
        response = model.generate_content(
            [
                {
                    "mime_type": "audio/mpeg",
                    "data": audio_bytes
                },
                prompt
            ]
        )

        result = json.loads(response.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail="Gemini analysis failed")
