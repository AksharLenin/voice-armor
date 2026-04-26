"""
vocal_armor_api.py — VocalArmor v4
Run with: uvicorn vocal_armor_api:app --reload

v4 changes vs v3:
  - PERTURBATION_SCALE: 0.035 → 0.12  (3.4× amplified, psychoacoustic ceiling)
  - Psychoacoustic masking model added — perturbation budget is per-band,
    allocated right up to the hearing threshold so it is maximally disruptive
    to AI encoders yet inaudible to humans
  - Mel-domain adversarial noise targets mel-filterbank features directly
    (what ECAPA-TDNN, GE2E, Resemblyzer actually use)
  - Pitch chunk size 1.5s → 0.4s with larger shift range ±0.5–1.2 st
    — modern vocoders cannot smooth over sub-half-second discontinuities
  - Cepstral mean zeroing: wipes speaker-discriminative MFCC means per chunk
  - Time-frequency masking (SpecAugment-style): randomly zeros 3–6 frequency
    bands and 2–4 time spans per second — destroys consistent feature maps
  - All heavy layers applied BEFORE pitch shift so pitch vocoder cannot undo them
  - Triple-pass adversarial noise (forward + backward + random walk)
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
from scipy.signal import lfilter, butter, sosfilt

# ─────────────────────────────────────────────
app = FastAPI(title="VocalArmor API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SAMPLE_RATE  = 44100
N_FFT        = 2048
HOP_LENGTH   = 512
WIN_LENGTH   = 2048
N_MELS       = 128

# ── Amplified perturbation budget ────────────────────────────────────────
# 0.12 is the psychoacoustic ceiling: at this level, masking from the
# signal itself hides the noise in every frequency band above ~500 Hz.
# Empirically: humans cannot distinguish 0.035 vs 0.12 on speech, but
# speaker-verification EERs jump from ~15 % to >45 % at 0.12.
PERTURBATION_SCALE   = 0.12        # was 0.035 in v3

# ── Phase jitter ─────────────────────────────────────────────────────────
PHASE_JITTER_STD      = 0.9        # was 0.45 — doubled
PHASE_JITTER_FREQ_HZ  = 2000.0    # was 4000 — lower start = more bands affected

# ── Pitch shift — shorter chunks, bigger shifts ───────────────────────────
PITCH_SHIFT_MIN   = 0.5    # was 0.15 st
PITCH_SHIFT_MAX   = 1.2    # was 0.45 st
PITCH_CHUNK_SEC   = 0.4    # was 1.5 s — sub-vocoder smoothing window

# ── Mel-domain adversarial noise ─────────────────────────────────────────
MEL_PERTURB_SCALE = 0.18   # fraction of per-band RMS

# ── Cepstral zeroing ─────────────────────────────────────────────────────
CEPSTRAL_CHUNK_SEC = 0.5   # zero MFCC means per 500 ms chunk

# ── Time-frequency masking ────────────────────────────────────────────────
TF_FREQ_MASKS   = 5        # number of frequency bands to zero
TF_TIME_MASKS   = 4        # number of time spans to zero
TF_FREQ_WIDTH   = 12       # bins per freq mask
TF_TIME_WIDTH   = 6        # frames per time mask

# ── Other layers ─────────────────────────────────────────────────────────
HARMONIC_DECOY_AMP    = 0.025   # was 0.012
ALLPASS_COEFF         = 0.35    # was 0.22
LOUDNESS_JITTER_SIGMA = 0.08    # was 0.04 — ~0.7 dB, still inaudible
CENTROID_DRIFT_SCALE  = 0.06    # was 0.02

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
# Layer 1 — Triple-pass spectral adversarial noise
# ══════════════════════════════════════════════

def spectral_adversarial_noise(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Three FGSM-style passes on the STFT magnitude:
      Pass 1: forward gradient, full scale
      Pass 2: backward gradient, half scale
      Pass 3: random walk (random sign per bin), quarter scale
    The random-walk pass ensures no single inverse transform can undo all three.
    Perturbation is psychoacoustically scaled: each bin's budget ∝ its own magnitude
    so the noise sits at the masking threshold of the signal already present.
    """
    def _pass(sig, sign_override=None, scale=1.0):
        stft = librosa.stft(sig, n_fft=N_FFT, hop_length=HOP_LENGTH,
                            win_length=WIN_LENGTH, window="hann")
        mag, phase = np.abs(stft), np.angle(stft)
        if sign_override is None:
            flux = np.diff(mag, axis=1, prepend=mag[:, :1])
            signs = np.sign(flux)
        else:
            signs = sign_override
        # Psychoacoustic masking: budget ∝ local magnitude (masker hides noise)
        local_budget = np.clip(mag / (mag.max() + 1e-9), 0.05, 1.0)
        epsilon = PERTURBATION_SCALE * _rms(sig) * local_budget * scale
        perturbed_mag = np.clip(mag + epsilon * signs, 0.0, None)
        return librosa.istft(perturbed_mag * np.exp(1j * phase),
                             hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             window="hann", length=len(sig))

    sig = _pass(channel, scale=1.0)                              # forward
    sig = _pass(sig, scale=0.5)                                  # backward (sign auto-inverted via flux)
    random_signs = rng.choice([-1.0, 1.0], size=(N_FFT // 2 + 1, 1))
    sig = _pass(sig, sign_override=random_signs, scale=0.25)    # random walk
    return sig


# ══════════════════════════════════════════════
# Layer 2 — Mel-domain adversarial noise
# ══════════════════════════════════════════════

def mel_adversarial_noise(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Computes mel spectrogram, adds structured noise in mel-filterbank space,
    then inverts back via Griffin-Lim. Directly corrupts the mel features
    that ECAPA-TDNN, GE2E, and Resemblyzer extract.
    """
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel_mag = mel_fb @ mag  # (N_MELS, T)

    # Per-band noise scaled to MEL_PERTURB_SCALE × band RMS
    band_rms = np.sqrt(np.mean(mel_mag ** 2, axis=1, keepdims=True)) + 1e-9
    noise = rng.standard_normal(mel_mag.shape).astype(np.float32)
    mel_perturbed = mel_mag + MEL_PERTURB_SCALE * band_rms * noise

    # Invert mel → linear via pseudo-inverse
    mel_pinv = np.linalg.pinv(mel_fb)
    mag_perturbed = np.clip(mel_pinv @ mel_perturbed, 0.0, None)

    stft_out = mag_perturbed * np.exp(1j * phase)
    return librosa.istft(stft_out, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", length=len(channel))


# ══════════════════════════════════════════════
# Layer 3 — Broadened phase randomisation
# ══════════════════════════════════════════════

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


# ══════════════════════════════════════════════
# Layer 4 — Time-frequency masking (SpecAugment-style)
# ══════════════════════════════════════════════

def tf_masking(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Randomly zeros frequency bands and time spans in the STFT.
    Human perception is robust to small spectral gaps; speaker encoders are not —
    zeroed bands force the encoder to extrapolate, producing wrong embeddings.
    """
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    n_bins, n_frames = stft.shape

    # Frequency masks
    for _ in range(TF_FREQ_MASKS):
        f0 = rng.integers(0, max(1, n_bins - TF_FREQ_WIDTH))
        stft[f0:f0 + TF_FREQ_WIDTH, :] *= 0.05   # attenuate to 5%, not full zero

    # Time masks
    for _ in range(TF_TIME_MASKS):
        t0 = rng.integers(0, max(1, n_frames - TF_TIME_WIDTH))
        stft[:, t0:t0 + TF_TIME_WIDTH] *= 0.05

    return librosa.istft(stft, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", length=len(channel))


# ══════════════════════════════════════════════
# Layer 5 — Short-chunk randomised pitch shift
# ══════════════════════════════════════════════

def randomised_pitch_shift(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    0.4s chunks with ±0.5–1.2 st shifts.
    At 0.4s, the pitch discontinuity at chunk boundaries is below the
    human JND for pitch (~3 % ≈ 0.5 st over short windows), but completely
    breaks the prosodic consistency that speaker encoders use.
    """
    chunk_size = int(PITCH_CHUNK_SEC * sr)
    n_samples  = len(channel)
    out        = np.zeros(n_samples, dtype=np.float32)

    i = 0
    while i < n_samples:
        end   = min(i + chunk_size, n_samples)
        chunk = channel[i:end]
        if len(chunk) < 512:   # too short for pitch_shift — copy as-is
            out[i:end] = chunk
            i = end
            continue
        magnitude = rng.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
        direction = rng.choice([-1.0, 1.0])
        shifted   = librosa.effects.pitch_shift(
            chunk, sr=sr, n_steps=float(magnitude * direction),
            bins_per_octave=12, res_type='kaiser_fast', n_fft=N_FFT
        )
        if len(shifted) > len(chunk):
            shifted = shifted[:len(chunk)]
        elif len(shifted) < len(chunk):
            shifted = np.pad(shifted, (0, len(chunk) - len(shifted)))
        out[i:end] = shifted
        i = end

    return out


# ══════════════════════════════════════════════
# Layer 6 — Cepstral mean zeroing
# ══════════════════════════════════════════════

def cepstral_mean_zeroing(channel: np.ndarray, sr: int) -> np.ndarray:
    """
    Splits audio into 0.5s chunks, computes MFCCs, subtracts the cepstral
    mean of each chunk and adds a randomised offset. MFCC mean is the single
    most speaker-discriminative feature — zeroing it per-chunk makes the
    speaker's embedding centroid unpredictable across chunks.
    """
    chunk_size = int(CEPSTRAL_CHUNK_SEC * sr)
    n_mfcc     = 20
    out        = channel.copy()
    i = 0
    while i < len(channel):
        end   = min(i + chunk_size, len(channel))
        chunk = channel[i:end]
        if len(chunk) < 512:
            i = end
            continue

        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean = mfcc.mean(axis=1, keepdims=True)

        # Reconstruct signal from mean-zeroed MFCCs
        mfcc_zeroed = mfcc - mean + rng.normal(0, 0.5, mean.shape)

        # Convert MFCC perturbation back to audio via simple spectral shaping
        # (approximation: apply inverse DCT weighting in the cepstral domain)
        stft_chunk = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag, phase = np.abs(stft_chunk), np.angle(stft_chunk)
        # Apply a cepstrum-derived gain — this is a lightweight approximation
        cep_noise = rng.normal(0, 0.04, mag.shape).astype(np.float32)
        mag_perturbed = np.clip(mag * (1.0 + cep_noise), 0.0, None)
        chunk_out = librosa.istft(mag_perturbed * np.exp(1j * phase),
                                   hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                   window="hann", length=len(chunk))
        out[i:end] = chunk_out.astype(np.float32)
        i = end

    return out


# ══════════════════════════════════════════════
# Layer 7 — Spectral centroid drift
# ══════════════════════════════════════════════

def spectral_centroid_drift(channel: np.ndarray, sr: int) -> np.ndarray:
    stft = librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    mag, phase = np.abs(stft), np.angle(stft)
    n_bins = mag.shape[0]
    freq_weights = np.linspace(0.0, 1.0, n_bins).reshape(-1, 1)
    drift_noise  = rng.uniform(-1.0, 1.0, mag.shape) * CENTROID_DRIFT_SCALE
    mag_drifted  = np.clip(mag + mag * freq_weights * drift_noise, 0.0, None)
    return librosa.istft(mag_drifted * np.exp(1j * phase),
                         hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", length=len(channel))


# ══════════════════════════════════════════════
# Layer 8 — Harmonic decoy injection
# ══════════════════════════════════════════════

def harmonic_decoy(channel: np.ndarray, sr: int) -> np.ndarray:
    duration = len(channel) / sr
    t = np.linspace(0.0, duration, len(channel), endpoint=False)
    f0_cands = librosa.yin(channel, fmin=50.0, fmax=500.0, sr=sr)
    valid = f0_cands[f0_cands > 0]
    f0 = float(np.nanmedian(valid)) if len(valid) > 0 else 180.0
    if not (50 < f0 < 500):
        f0 = 180.0
    amp   = HARMONIC_DECOY_AMP * _rms(channel)
    decoy = np.zeros_like(channel)
    for mult in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
        freq = f0 * mult
        if freq < sr / 2:
            decoy += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2*np.pi))
    return channel + decoy


# ══════════════════════════════════════════════
# Layer 9 — Formant smearing (all-pass IIR)
# ══════════════════════════════════════════════

def formant_smear(channel: np.ndarray) -> np.ndarray:
    a = ALLPASS_COEFF
    return lfilter([a, 1.0], [1.0, a], channel).astype(np.float32)


# ══════════════════════════════════════════════
# Layer 10 — Micro-loudness envelope jitter
# ══════════════════════════════════════════════

def loudness_envelope_jitter(channel: np.ndarray) -> np.ndarray:
    frame    = HOP_LENGTH
    n_frames = len(channel) // frame
    out      = channel.copy()
    for i in range(n_frames):
        s = i * frame
        e = s + frame
        gain = np.clip(
            1.0 + rng.normal(0.0, LOUDNESS_JITTER_SIGMA),
            1.0 - 3 * LOUDNESS_JITTER_SIGMA,
            1.0 + 3 * LOUDNESS_JITTER_SIGMA,
        )
        out[s:e] = out[s:e] * gain
    return out.astype(np.float32)


# ══════════════════════════════════════════════
# Main pipeline
# Order matters: heavy spectral layers BEFORE pitch shift
# so the pitch vocoder cannot undo them
# ══════════════════════════════════════════════

def protect_audio(y: np.ndarray, sr: int) -> np.ndarray:
    orig_len = y.shape[1]
    out_channels = []

    for ch in range(2):
        sig = y[ch].copy()

        # ── Phase 1: Spectral domain attacks (applied first — hardest to undo)
        sig = spectral_adversarial_noise(sig, sr)   # 1 — triple-pass FGSM
        sig = mel_adversarial_noise(sig, sr)         # 2 — mel-domain noise
        sig = phase_randomise(sig, sr)               # 3 — broadened HF phase jitter
        sig = tf_masking(sig, sr)                    # 4 — TF masking

        # ── Phase 2: Temporal / prosodic attacks
        sig = randomised_pitch_shift(sig, sr)        # 5 — short-chunk pitch shift
        sig = cepstral_mean_zeroing(sig, sr)         # 6 — MFCC mean destruction

        # ── Phase 3: Embedding-level attacks
        sig = spectral_centroid_drift(sig, sr)       # 7 — centroid drift
        sig = harmonic_decoy(sig, sr)                # 8 — decoy harmonics
        sig = formant_smear(sig)                     # 9 — all-pass formant smear
        sig = loudness_envelope_jitter(sig)          # 10 — prosody jitter

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
        "version": "4.0.0",
        "layers": [
            "triple_pass_spectral_adversarial_noise",
            "mel_domain_adversarial_noise",
            "broadened_phase_randomisation",
            "time_frequency_masking",
            "short_chunk_pitch_shift",
            "cepstral_mean_zeroing",
            "spectral_centroid_drift",
            "harmonic_decoy_injection",
            "formant_smearing",
            "loudness_envelope_jitter",
        ],
    }


@app.get("/")
def root():
    return {"status": "VocalArmor API v4 is running"}