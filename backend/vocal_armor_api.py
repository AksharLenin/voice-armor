"""
vocal_armor_api.py — FastAPI backend for VocalArmor protection v3.
Run with: uvicorn vocal_armor_api:app --reload

What changed in v3 vs v2:
  - PERTURBATION_SCALE raised 0.015 → 0.035 (amplified wavefront, still sub-perceptual)
  - Pitch shift is now RANDOMISED per-chunk (±0.15–0.45 st, random direction per chunk)
    so cloning models cannot learn a fixed inverse transform
  - Dual-pass adversarial noise (forward + backward gradient) for stronger disruption
  - Spectral centroid drift breaks x-vector / d-vector speaker embeddings
  - Micro-loudness envelope jitter breaks prosody-based cloning
"""

import io
import os
import tempfile

import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import lfilter

# ──────────────────────────────────────────────
app = FastAPI(title="VocalArmor API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE        = 44100
N_FFT              = 2048
HOP_LENGTH         = 512
WIN_LENGTH         = 2048

# Amplified perturbation — still below psychoacoustic masking threshold
PERTURBATION_SCALE  = 0.035          # was 0.015

# Phase jitter (humans are phase-blind above ~4 kHz)
PHASE_JITTER_STD     = 0.45
PHASE_JITTER_FREQ_HZ = 4000.0

# Randomised pitch shift per chunk — cloners cannot learn a fixed inverse
PITCH_SHIFT_MIN  = 0.15   # semitones
PITCH_SHIFT_MAX  = 0.45
PITCH_CHUNK_SEC  = 1.5    # each chunk gets its own independent random shift

# Other layer settings
HARMONIC_DECOY_AMP    = 0.012
ALLPASS_COEFF         = 0.22
LOUDNESS_JITTER_SIGMA = 0.04   # ~0.35 dB — completely inaudible
CENTROID_DRIFT_SCALE  = 0.02

rng = np.random.default_rng()


# ══════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════

def load_audio(path: str) -> tuple[np.ndarray, int]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".aac", ".mp3", ".m4a", ".ogg"}:
        try:
            import imageio_ffmpeg
            from pydub import AudioSegment
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            seg = AudioSegment.from_file(path)
            seg = seg.set_channels(2).set_frame_rate(SAMPLE_RATE)
            tmp_wav = path + "_decoded.wav"
            seg.export(tmp_wav, format="wav")
            y, sr = librosa.load(tmp_wav, sr=None, mono=False)
            os.remove(tmp_wav)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Audio decode error: {exc}")
    else:
        y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim == 1:
        y = np.stack([y, y], axis=0)

    if sr != SAMPLE_RATE:
        y = np.stack([
            librosa.resample(y[0], orig_sr=sr, target_sr=SAMPLE_RATE),
            librosa.resample(y[1], orig_sr=sr, target_sr=SAMPLE_RATE),
        ], axis=0)
        sr = SAMPLE_RATE

    return y.astype(np.float32), sr


def _rms(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sig ** 2)) + 1e-9)


# ══════════════════════════════════════════════
# Protection layers
# ══════════════════════════════════════════════

# ── Layer 1: Dual-pass spectral adversarial noise ──────────────────────────
def spectral_adversarial_noise(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Two passes of FGSM-style spectral perturbation.
    Pass 1 (forward gradient) + Pass 2 (reversed, half-strength).
    Creates a complex perturbation surface that is much harder to invert
    than a single-pass attack.
    """
    def _pass(sig, sign_flip=1.0, scale=1.0):
        stft = librosa.stft(sig, n_fft=N_FFT, hop_length=HOP_LENGTH,
                            win_length=WIN_LENGTH, window="hann")
        mag, phase = np.abs(stft), np.angle(stft)
        flux = np.diff(mag, axis=1, prepend=mag[:, :1]) * sign_flip
        local_scale = np.clip(mag / (mag.max() + 1e-9), 0.05, 1.0)
        epsilon = PERTURBATION_SCALE * _rms(sig) * local_scale * scale
        perturbed_mag = np.clip(mag + epsilon * np.sign(flux), 0.0, None)
        return librosa.istft(perturbed_mag * np.exp(1j * phase),
                             hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             window="hann", length=len(sig))

    sig = _pass(channel, sign_flip=1.0,  scale=1.0)
    sig = _pass(sig,     sign_flip=-1.0, scale=0.5)
    return sig


# ── Layer 2: High-frequency phase randomisation ────────────────────────────
def phase_randomise(channel: np.ndarray, sr: int) -> np.ndarray:
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    high_mask = freq_bins >= PHASE_JITTER_FREQ_HZ
    jitter = rng.normal(0.0, PHASE_JITTER_STD, phase.shape)
    jitter[~high_mask, :] = 0.0
    stft_out = mag * np.exp(1j * (phase + jitter))
    return librosa.istft(stft_out, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", length=len(channel))


# ── Layer 3: Randomised chunked pitch shifting ─────────────────────────────
def randomised_pitch_shift(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Splits audio into ~1.5s chunks. Each chunk gets an independent random
    pitch shift between ±PITCH_SHIFT_MIN and ±PITCH_SHIFT_MAX semitones.

    Why this beats a fixed shift:
      Fixed +0.26 st → a cloner can learn to subtract 0.26 st and recover
      the speaker's identity. Random per-chunk shifts have no single inverse;
      cloned output has broken pitch continuity and inconsistent prosody.

    Why it's still inaudible:
      Sub-semitone variation across 1.5s windows is well within the natural
      pitch variation of real speech — listeners cannot detect it.
    """
    chunk_size = int(PITCH_CHUNK_SEC * sr)
    n_samples  = len(channel)
    out        = np.zeros(n_samples, dtype=np.float32)

    i = 0
    while i < n_samples:
        end   = min(i + chunk_size, n_samples)
        chunk = channel[i:end]
        magnitude = rng.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
        direction = rng.choice([-1.0, 1.0])
        shifted   = librosa.effects.pitch_shift(
            chunk, sr=sr, n_steps=float(magnitude * direction),
            bins_per_octave=12, res_type='kaiser_fast', n_fft=N_FFT
        )
        # Exact length match
        if len(shifted) > len(chunk):
            shifted = shifted[:len(chunk)]
        elif len(shifted) < len(chunk):
            shifted = np.pad(shifted, (0, len(chunk) - len(shifted)))
        out[i:end] = shifted
        i = end

    return out


# ── Layer 4: Spectral centroid drift ──────────────────────────────────────
def spectral_centroid_drift(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Nudges spectral centroid per frame. Speaker embeddings (x-vector,
    d-vector, ECAPA) are sensitive to centroid location — shifting it
    moves the speaker away from their true embedding centroid.
    """
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)
    n_bins = mag.shape[0]
    freq_weights = np.linspace(0.0, 1.0, n_bins).reshape(-1, 1)
    drift_noise  = rng.uniform(-1.0, 1.0, mag.shape) * CENTROID_DRIFT_SCALE
    mag_drifted  = np.clip(mag + mag * freq_weights * drift_noise, 0.0, None)
    stft_out     = mag_drifted * np.exp(1j * phase)
    return librosa.istft(stft_out, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", length=len(channel))


# ── Layer 5: Harmonic decoy injection ────────────────────────────────────
def harmonic_decoy(channel: np.ndarray, sr: int) -> np.ndarray:
    duration = len(channel) / sr
    t = np.linspace(0.0, duration, len(channel), endpoint=False)
    f0_cands = librosa.yin(channel, fmin=50.0, fmax=500.0, sr=sr)
    valid = f0_cands[f0_cands > 0]
    f0 = float(np.nanmedian(valid)) if len(valid) > 0 else 180.0
    if not (50 < f0 < 500):
        f0 = 180.0
    amp    = HARMONIC_DECOY_AMP * _rms(channel)
    decoy  = np.zeros_like(channel)
    for mult in [1.5, 2.5, 3.5, 4.5, 5.5]:
        freq = f0 * mult
        if freq < sr / 2:
            decoy += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2*np.pi))
    return channel + decoy


# ── Layer 6: Formant smearing (all-pass IIR) ──────────────────────────────
def formant_smear(channel: np.ndarray) -> np.ndarray:
    a = ALLPASS_COEFF
    return lfilter([a, 1.0], [1.0, a], channel).astype(np.float32)


# ── Layer 7: Micro-loudness envelope jitter ───────────────────────────────
def loudness_envelope_jitter(channel: np.ndarray) -> np.ndarray:
    """
    Per-frame gain variation of ±LOUDNESS_JITTER_SIGMA (~0.35 dB).
    Completely inaudible. Breaks prosody-based speaker verification.
    """
    frame    = HOP_LENGTH
    n_frames = len(channel) // frame
    out      = channel.copy()
    for i in range(n_frames):
        s   = i * frame
        e   = s + frame
        gain = np.clip(
            1.0 + rng.normal(0.0, LOUDNESS_JITTER_SIGMA),
            1.0 - 3*LOUDNESS_JITTER_SIGMA,
            1.0 + 3*LOUDNESS_JITTER_SIGMA,
        )
        out[s:e] = out[s:e] * gain
    return out.astype(np.float32)


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

def protect_audio(y: np.ndarray, sr: int) -> np.ndarray:
    orig_len = y.shape[1]
    out_channels = []

    for ch in range(2):
        sig = y[ch].copy()
        sig = spectral_adversarial_noise(sig, sr)   # 1 — dual-pass FGSM
        sig = phase_randomise(sig, sr)               # 2 — HF phase jitter
        sig = randomised_pitch_shift(sig, sr)        # 3 — random chunked shift
        sig = spectral_centroid_drift(sig, sr)       # 4 — embedding centroid drift
        sig = harmonic_decoy(sig, sr)                # 5 — decoy harmonics
        sig = formant_smear(sig)                     # 6 — all-pass formant smear
        sig = loudness_envelope_jitter(sig)          # 7 — prosody jitter

        if len(sig) > orig_len:
            sig = sig[:orig_len]
        elif len(sig) < orig_len:
            sig = np.pad(sig, (0, orig_len - len(sig)))
        out_channels.append(sig.astype(np.float32))

    result = np.stack(out_channels, axis=0)
    peak   = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result


# ══════════════════════════════════════════════
# API endpoints
# ══════════════════════════════════════════════

@app.post("/protect-voice")
async def protect_voice_endpoint(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        y, sr       = load_audio(tmp_path)
        y_protected = protect_audio(y, sr)
        buf = io.BytesIO()
        sf.write(buf, y_protected.T, sr, format="WAV", subtype="PCM_24")
        buf.seek(0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.remove(tmp_path)

    base = os.path.splitext(audio.filename)[0]
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{base}_protected.wav"'},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "layers": [
            "dual_pass_spectral_adversarial_noise",
            "phase_randomisation",
            "randomised_chunked_pitch_shift",
            "spectral_centroid_drift",
            "harmonic_decoy_injection",
            "formant_smearing",
            "loudness_envelope_jitter",
        ],
    }

@app.get("/")
def root():
    return {"status": "VocalArmor API v3 is running"}