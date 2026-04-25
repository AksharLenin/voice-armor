"""
vocal_armor_api.py — FastAPI backend for vocal armor protection.
Run with: uvicorn vocal_armor_api:app --reload
"""

import numpy as np
import librosa
import soundfile as sf
import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PITCH_SHIFT = 0.26


def load_audio(input_file):
    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".aac", ".mp3", ".m4a", ".ogg"]:
        import subprocess
        temp_wav = input_file + "_decoded.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_file,
            "-ac", "2", "-ar", "44100",
            "-acodec", "pcm_s16le", temp_wav
        ], check=True, capture_output=True)
        y, sr = librosa.load(temp_wav, sr=None, mono=False)
        os.remove(temp_wav)
    else:
        y, sr = librosa.load(input_file, sr=None, mono=False)

    if y.ndim == 1:
        y = np.stack([y, y], axis=0)

    return y, sr


def protect_audio(y, sr):
    orig_len = y.shape[1]

    left = librosa.effects.pitch_shift(
        y[0], sr=sr, n_steps=PITCH_SHIFT,
        bins_per_octave=12, res_type='soxr_hq', n_fft=2048
    )
    right = librosa.effects.pitch_shift(
        y[1], sr=sr, n_steps=PITCH_SHIFT,
        bins_per_octave=12, res_type='soxr_hq', n_fft=2048
    )

    # Pad or trim to exactly match original length
    def fix_length(sig, length):
        if len(sig) > length:
            return sig[:length]
        return np.pad(sig, (0, length - len(sig)))

    y_shifted = np.stack([
        fix_length(left,  orig_len),
        fix_length(right, orig_len)
    ], axis=0)

    # Normalize
    peak = np.max(np.abs(y_shifted))
    if peak > 1.0:
        y_shifted = y_shifted / peak

    return y_shifted.astype(np.float32)


@app.post("/protect-voice")
async def protect(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        y, sr       = load_audio(tmp_path)
        y_protected = protect_audio(y, sr)

        buf = io.BytesIO()
        sf.write(buf, y_protected.T, sr, format="WAV")
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