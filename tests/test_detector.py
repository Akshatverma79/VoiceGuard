"""
VoiceGuard — tests/test_detector.py
Local test script for the AASIST anti-spoofing detector.

Usage:
    python tests/test_detector.py data/real/sample.wav
    python tests/test_detector.py data/fake/spoof.wav

The result displays:
    Audio:
    Prediction:
    Spoof Probability:
    Inference Time:

All values come from actual AASIST model inference.
No fake or random values are used.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Make sure ml/ is on the path ──────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
_ML_DIR = _ROOT / "ml"
sys.path.insert(0, str(_ML_DIR))
sys.path.insert(0, str(_ROOT))


def run_test(audio_path: str) -> None:
    """Run AASIST inference and print the result to stdout."""
    print("=" * 55)
    print("  VoiceGuard — AASIST Anti-Spoofing Test")
    print("=" * 55)

    audio_path = str(Path(audio_path).resolve())

    # ── Import detector ───────────────────────────────────────────────────
    print("\n[1/3] Loading AASIST model…")
    try:
        from detector import AASISTDetector
    except ImportError as exc:
        print(f"\n❌ Import error: {exc}")
        print(
            "\nMake sure you have installed the ML requirements:\n"
            "  pip install torch torchaudio numpy\n"
        )
        sys.exit(1)

    t_load_start = time.perf_counter()
    try:
        detector = AASISTDetector()
    except RuntimeError as exc:
        print(f"\n❌ Model loading failed:\n{exc}")
        sys.exit(1)
    t_load_end = time.perf_counter()

    print(f"    Model loaded in {t_load_end - t_load_start:.2f}s\n")

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"[2/3] Running inference on: {audio_path}")
    try:
        result = detector.predict(audio_path)
    except FileNotFoundError as exc:
        print(f"\n❌ File not found:\n{exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n❌ Invalid audio:\n{exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n❌ Inference error:\n{exc}")
        sys.exit(1)

    # ── Display result ────────────────────────────────────────────────────
    print("\n[3/3] Result:")
    print("-" * 55)
    print(f"  Audio            : {Path(audio_path).name}")
    print(f"  Prediction       : {result['prediction'].upper()}")
    print(f"  Spoof Probability: {result['spoof_probability']:.4f}  ({result['spoof_probability']*100:.2f}%)")
    print(f"  Real  Probability: {result['real_probability']:.4f}  ({result['real_probability']*100:.2f}%)")
    print(f"  Inference Time   : {result['inference_time_s']:.4f} seconds")
    print(f"  Device           : {result['device']}")
    print("-" * 55)

    # ── Visual verdict ────────────────────────────────────────────────────
    if result["prediction"] == "spoof":
        print("\n  🚨 VERDICT: SPOOF DETECTED — This audio may be a deepfake.")
    else:
        print("\n  ✅ VERDICT: REAL — No spoofing detected.")

    print("=" * 55 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoiceGuard Phase 1 — AASIST anti-spoofing test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_detector.py data/real/sample.wav
  python tests/test_detector.py data/fake/spoof.wav
  python tests/test_detector.py /absolute/path/to/audio.wav
        """,
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the WAV audio file to analyze.",
    )
    args = parser.parse_args()
    run_test(args.audio_path)


if __name__ == "__main__":
    main()
