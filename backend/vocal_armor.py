"""
vocal_armor.py — Run this directly to protect your voice recording.
"""

import numpy as np
import librosa
import soundfile as sf
import os
import glob


# ═══════════════════════════════════════════════════════════════
#  ✏  SET YOUR OUTPUT FOLDER
# ═══════════════════════════════════════════════════════════════
OUTPUT_FOLDER = r"C:\Users\Akku\OneDrive\Desktop\Voice"

# ═══════════════════════════════════════════════════════════════
#  Settings
# ═══════════════════════════════════════════════════════════════
PITCH_SHIFT = 0.2   # semitones (0.3–1.0 recommended, negative = lower pitch)


def load_audio(input_file):
    """Load audio file, converting from AAC/MP3/OGG/M4A if needed."""
    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".aac", ".mp3", ".m4a", ".ogg"]:
        from pydub import AudioSegment
        print(f"  Converting {ext} to wav...")
        audio_seg = AudioSegment.from_file(input_file)
        temp_wav  = input_file.rsplit(".", 1)[0] + "_temp.wav"
        audio_seg.export(temp_wav, format="wav")
        y, sr = librosa.load(temp_wav, sr=None, mono=False)
        os.remove(temp_wav)
    else:
        y, sr = librosa.load(input_file, sr=None, mono=False)
    return y, sr


def protect_audio(y, sr):
    """Pitch shift + noise reduction."""
    # Pitch shift
    if y.ndim == 2:
        left  = librosa.effects.pitch_shift(
            y[0], sr=sr, n_steps=PITCH_SHIFT,
            bins_per_octave=12, res_type='soxr_hq'
        )
        right = librosa.effects.pitch_shift(
            y[1], sr=sr, n_steps=PITCH_SHIFT,
            bins_per_octave=12, res_type='soxr_hq'
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


if __name__ == "__main__":
    print("=" * 50)
    print("  🛡️  VOCAL-ARMOR  —  Voice Clone Protection")
    print("=" * 50)

    # Auto-detect audio files in the same folder as the script
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    supported   = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aac"]
    audio_files = []
    for pattern in supported:
        audio_files.extend(glob.glob(os.path.join(script_dir, pattern)))

    if not audio_files:
        print(f"\n  ❌ No audio file found in:\n     {script_dir}")
        print(f"\n  👉 Put your audio file (wav/mp3/flac/ogg/m4a/aac) in the same folder as this script.")
        input("\n  Press Enter to exit...")
        raise SystemExit

    if len(audio_files) > 1:
        print(f"\n  Found multiple audio files:")
        for i, f in enumerate(audio_files):
            print(f"    [{i+1}] {os.path.basename(f)}")
        choice = input("\n  Enter number to select: ").strip()
        try:
            INPUT_FILE = audio_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice, using first file.")
            INPUT_FILE = audio_files[0]
    else:
        INPUT_FILE = audio_files[0]
        print(f"\n  Found: {os.path.basename(INPUT_FILE)}")

    # Build output filename
    input_basename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    OUTPUT_FILE    = os.path.join(OUTPUT_FOLDER, f"{input_basename}_protected.wav")

    print(f"\n  📂 Input  : {INPUT_FILE}")
    print(f"  💾 Output : {OUTPUT_FILE}")

    # Load
    print("\n  Loading audio...")
    y, sr = load_audio(INPUT_FILE)
    print(f"  Sample rate : {sr} Hz")
    print(f"  Channels    : {'Stereo' if y.ndim == 2 else 'Mono'}")

    # Protect
    print(f"  Applying pitch shift ({PITCH_SHIFT:+.1f} semitones)...")
    y_protected = protect_audio(y, sr)

    # Save
    print("\n  Saving output...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if y_protected.ndim == 2:
        sf.write(OUTPUT_FILE, y_protected.T, sr)
    else:
        sf.write(OUTPUT_FILE, y_protected, sr)

    print("\n" + "=" * 50)
    print("  ✅  DONE!")
    print("=" * 50)
    print(f"  Input file  : {os.path.basename(INPUT_FILE)}")
    print(f"  Duration    : {y.shape[-1]/sr:.1f}s")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Pitch shift : {PITCH_SHIFT:+.1f} semitones")
    print(f"\n  Output saved to:\n  {OUTPUT_FILE}")
    print("=" * 50)

    input("\n  Press Enter to exit...")
