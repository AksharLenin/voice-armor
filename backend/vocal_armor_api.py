"""
vocal_armor_api.py — FastAPI backend for vocal armor protection.
Run with: uvicorn vocal_armor_api:app --reload
"""

import numpy as np
import librosa
import soundfile as sf
import os
import glob
import io
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
#  Settings  (unchanged from your script)
# ═══════════════════════════════════════════════════════════════
PITCH_SHIFT = 0.3   # semitones (0.3–1.0 recommended, negative = lower pitch)


def load_audio(input_file):
    """Load audio file, converting from AAC/MP3/OGG/M4A if needed."""
    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".aac", ".mp3", ".m4a", ".ogg"]:
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(input_file)
        temp_wav  = input_file.rsplit(".", 1)[0] + "_temp.wav"
        audio_seg.export(temp_wav, format="wav")
        y, sr = librosa.load(temp_wav, sr=None, mono=False)
        os.remove(temp_wav)
    else:
        y, sr = librosa.load(input_file, sr=None, mono=False)
    return y, sr


def protect_audio(y, sr):
    """Pitch shift — exact same logic as vocal_armor.py."""
    if y.ndim == 2:
        left  = librosa.effects.pitch_shift(
            y[0], sr=sr, n_steps=PITCH_SHIFT,
            bins_per_octave=12, res_type='soxr_hq', n_fft=2048
        )
        right = librosa.effects.pitch_shift(
            y[1], sr=sr, n_steps=PITCH_SHIFT,
            bins_per_octave=12, res_type='soxr_hq', n_fft=2048
        )
        y_shifted = np.stack([left, right], axis=0)
    else: 
        y_shifted = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=PITCH_SHIFT,
            bins_per_octave=12, res_type='soxr_hq'
        )

    # Normalize
    peak = np.max(np.abs(y_shifted))
    if peak > 1.0:
        y_shifted = y_shifted / peak

    return y_shifted


@app.post("/protect-voice")
async def protect(audio: UploadFile = File(...)):
    # Save uploaded file to a temp file
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        y, sr = load_audio(tmp_path)
        y_protected = protect_audio(y, sr)

        # Write to memory buffer
        buf = io.BytesIO()
        if y_protected.ndim == 2:
            sf.write(buf, y_protected.T, sr, format="WAV")
        else:
            sf.write(buf, y_protected, sr, format="WAV")
        buf.seek(0)

    finally:
        os.remove(tmp_path)

    original_name = os.path.splitext(audio.filename)[0]
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{original_name}_protected.wav"'
        }
    )


@app.get("/")
def root():
    return {"status": "Vocal Armor API is running"}