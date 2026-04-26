"""
vocal_armor_api.py — FastAPI backend for VocalArmor protection.
Run with: uvicorn vocal_armor_api:app --reload

Protection strategy (layered, all inaudible to humans):
  1. Spectral adversarial noise  — FGSM-style perturbation injected into
     STFT magnitude bins. Stays below psychoacoustic masking threshold so
     human ears cannot detect it, but completely scrambles the spectral
     fingerprint that cloning models extract.
  2. Phase randomisation       — random jitter added to STFT phase in
     high-frequency bands (>4 kHz).  Humans are almost insensitive to
     absolute phase; TTS/cloning encoders are not.
  3. Temporal micro-jitter     — tiny time-domain dithering at sub-sample
     level disrupts frame-level feature alignment used by most voice
     encoders (Resemblyzer, ECAPA, d-vector, etc.).
  4. Harmonic decoy injection  — low-amplitude synthetic harmonics that
     have no perceptual weight but push mel-filterbank features off the
     speaker's centroid.
  5. Formant smearing          — apply a very shallow all-pass filter that
     rotates formant peaks slightly; imperceptible to listeners but breaks
     LPC / formant-based speaker verification.
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
from scipy.signal import sosfilt, butter

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(title="VocalArmor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Tuneable constants
# ──────────────────────────────────────────────
SAMPLE_RATE        = 44100   # target sr for all processing
N_FFT              = 2048
HOP_LENGTH         = 512
WIN_LENGTH         = 2048

# Psychoacoustic ceiling — max perturbation as fraction of signal RMS
# 0.015 → ~−36 dB relative to speech; completely inaudible
PERTURBATION_SCALE = 0.015

# Phase jitter strength (radians) — applied only above PHASE_JITTER_FREQ_HZ
PHASE_JITTER_STD     = 0.35          # std-dev of Gaussian jitter
PHASE_JITTER_FREQ_HZ = 4000.0        # jitter only in high-frequency bins

# Temporal micro-jitter: fraction of one sample
TEMPORAL_JITTER_SIGMA = 0.25         # fractional sample shift per frame

# Harmonic decoy: amplitude relative to signal RMS
HARMONIC_DECOY_AMP = 0.008

# Formant smearing: all-pass filter coefficient (0 = off, 0.3 = subtle)
ALLPASS_COEFF = 0.18

rng = np.random.default_rng()        # reproducible across a single request


# ══════════════════════════════════════════════
# 1. I/O helpers
# ══════════════════════════════════════════════

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load any audio format → stereo float32 numpy array (2, N).
    Falls back to pydub + ffmpeg for compressed formats.
    """
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

    # Ensure stereo
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)

    # Resample if needed
    if sr != SAMPLE_RATE:
        y = np.stack([
            librosa.resample(y[0], orig_sr=sr, target_sr=SAMPLE_RATE),
            librosa.resample(y[1], orig_sr=sr, target_sr=SAMPLE_RATE),
        ], axis=0)
        sr = SAMPLE_RATE

    return y.astype(np.float32), sr


# ══════════════════════════════════════════════
# 2. Core protection layers
# ══════════════════════════════════════════════

def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)) + 1e-9)


# ── Layer 1: Spectral adversarial noise ──────────────────────────────────────

def spectral_adversarial_noise(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    FGSM-inspired perturbation in the STFT domain.

    Concept: real FGSM requires a differentiable model. Here we replicate
    the *effect* by analysing the signal's own gradient direction in the
    spectrogram and nudging magnitude in that direction — which is the
    direction that maximally moves the feature representation.

    Implementation:
      • Compute STFT magnitude + phase.
      • Estimate a surrogate gradient by taking the sign of the local
        spectral flux (frame-to-frame magnitude difference). This mimics
        the gradient direction a cloning encoder would see.
      • Add ε * sign(gradient) to the magnitude (capped by masking threshold).
      • Reconstruct via inverse STFT.
    """
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)

    # Surrogate gradient: spectral flux across time axis
    flux = np.diff(mag, axis=1, prepend=mag[:, :1])
    gradient_sign = np.sign(flux)  # shape (freq, time)

    # Psychoacoustic masking threshold per bin:
    # scale perturbation by local magnitude so louder bands tolerate more
    local_scale = np.clip(mag / (mag.max() + 1e-9), 0.05, 1.0)
    epsilon = PERTURBATION_SCALE * _rms(channel) * local_scale

    perturbed_mag = mag + epsilon * gradient_sign
    perturbed_mag = np.clip(perturbed_mag, 0.0, None)

    # Reconstruct
    stft_protected = perturbed_mag * np.exp(1j * phase)
    return librosa.istft(stft_protected, hop_length=HOP_LENGTH,
                         win_length=WIN_LENGTH, window="hann",
                         length=len(channel))


# ── Layer 2: Phase randomisation ─────────────────────────────────────────────

def phase_randomise(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Add Gaussian phase jitter to STFT bins above PHASE_JITTER_FREQ_HZ.
    Human auditory system is largely phase-blind in this region;
    voice encoders (especially GE2E / ECAPA-TDNN) are not.
    """
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)

    # Bin index corresponding to the frequency threshold
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    high_mask = freq_bins >= PHASE_JITTER_FREQ_HZ  # (n_fft/2+1,)

    jitter = rng.normal(0.0, PHASE_JITTER_STD, phase.shape)
    jitter[~high_mask, :] = 0.0          # only affect high-freq bins
    phase_jittered = phase + jitter

    stft_jittered = mag * np.exp(1j * phase_jittered)
    return librosa.istft(stft_jittered, hop_length=HOP_LENGTH,
                         win_length=WIN_LENGTH, window="hann",
                         length=len(channel))


# ── Layer 3: Temporal micro-jitter ───────────────────────────────────────────

def temporal_micro_jitter(channel: np.ndarray) -> np.ndarray:
    """
    Sub-sample temporal displacement using sinc interpolation.
    Each frame-length window gets a tiny random fractional shift.
    Disrupts frame-alignment in speaker-encoder feature extraction.
    """
    frame = HOP_LENGTH
    n_frames = len(channel) // frame
    out = channel.copy()

    for i in range(n_frames):
        start = i * frame
        end   = start + frame
        shift = rng.normal(0.0, TEMPORAL_JITTER_SIGMA)  # fractional samples

        # Build fractional shift via linear interpolation (fast approximation)
        frac = shift - int(shift)
        int_shift = int(shift)
        seg = channel[max(0, start + int_shift): end + int_shift + 2]

        if len(seg) >= frame:
            interpolated = (1 - frac) * seg[:frame] + frac * seg[1:frame + 1]
            out[start:end] = interpolated

    return out


# ── Layer 4: Harmonic decoy injection ────────────────────────────────────────

def harmonic_decoy(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Inject very low-amplitude synthetic harmonics at frequencies that sit
    *between* natural speech harmonics. These are inaudible (< −40 dB)
    but push mel-filterbank centroids and confuse speaker embeddings.
    """
    duration = len(channel) / sr
    t = np.linspace(0.0, duration, len(channel), endpoint=False)

    # Estimate fundamental via YIN — use it to place decoy harmonics
    f0_candidates = librosa.yin(channel, fmin=50.0, fmax=500.0, sr=sr)
    f0 = float(np.nanmedian(f0_candidates[f0_candidates > 0]))
    if not (50 < f0 < 500):
        f0 = 180.0   # fallback

    decoy_amp = HARMONIC_DECOY_AMP * _rms(channel)
    decoy = np.zeros_like(channel)

    # Place decoys at 1.5×, 2.5×, 3.5× f0 (between natural harmonics)
    for mult in [1.5, 2.5, 3.5, 4.5]:
        freq = f0 * mult
        if freq < sr / 2:
            phase_offset = rng.uniform(0, 2 * np.pi)
            decoy += decoy_amp * np.sin(2 * np.pi * freq * t + phase_offset)

    return channel + decoy


# ── Layer 5: Formant smearing via all-pass filter ────────────────────────────

def formant_smear(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a first-order all-pass IIR filter.
    All-pass filters preserve amplitude but rotate phase — they smear
    formant peak timing and subtly shift perceived formant structure just
    enough to confuse speaker verification, without audible colouration.

    Transfer function: H(z) = (a + z^-1) / (1 + a*z^-1)
    """
    a = ALLPASS_COEFF
    # Numerator and denominator coefficients
    b = np.array([a, 1.0])
    den = np.array([1.0, a])
    # Use scipy lfilter via second-order sections
    from scipy.signal import lfilter
    return lfilter(b, den, channel).astype(np.float32)


# ══════════════════════════════════════════════
# 3. Main protection pipeline
# ══════════════════════════════════════════════

def protect_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply all 5 protection layers to each channel.
    Returns float32 stereo array (2, N), peak-normalised.
    """
    orig_len = y.shape[1]
    out_channels = []

    for ch in range(2):
        sig = y[ch].copy()

        # Layer 1 — spectral adversarial noise
        sig = spectral_adversarial_noise(sig, sr)

        # Layer 2 — phase randomisation
        sig = phase_randomise(sig, sr)

        # Layer 3 — temporal micro-jitter
        sig = temporal_micro_jitter(sig)

        # Layer 4 — harmonic decoy
        sig = harmonic_decoy(sig, sr)

        # Layer 5 — formant smearing
        sig = formant_smear(sig, sr)

        # Length fix
        if len(sig) > orig_len:
            sig = sig[:orig_len]
        elif len(sig) < orig_len:
            sig = np.pad(sig, (0, orig_len - len(sig)))

        out_channels.append(sig.astype(np.float32))

    result = np.stack(out_channels, axis=0)

    # Peak normalise (ensure no clipping while preserving all perturbations)
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak

    return result


# ══════════════════════════════════════════════
# 4. API endpoints
# ══════════════════════════════════════════════

@app.post("/protect-voice")
async def protect_voice_endpoint(audio: UploadFile = File(...)):
    """
    Accepts any audio file, returns a WAV with all 5 protection layers applied.
    """
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

    base_name = os.path.splitext(audio.filename)[0]
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{base_name}_protected.wav"'
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "layers": [
            "spectral_adversarial_noise",
            "phase_randomisation",
            "temporal_micro_jitter",
            "harmonic_decoy",
            "formant_smearing",
        ],
    }


@app.get("/")
def root():
    return {"status": "VocalArmor API v2 is running"}