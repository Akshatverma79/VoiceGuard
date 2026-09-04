"""
VoiceGuard — tests/evaluate_audio.py
Local evaluation script: scans data/real/ and data/fake/, runs the full
Phase 2 pipeline on every WAV file, and reports per-file results.

If enough samples are present (≥ 2 total, ≥ 1 per class), also computes:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix

Usage (from VoiceGuard root):
    python tests/evaluate_audio.py

Place test files:
    data/real/person1.wav    ← genuine speech recordings
    data/fake/generated.wav  ← TTS / voice-clone samples

Important:
    - All metrics are computed on YOUR provided samples.
    - Do NOT interpret results from a small set as model accuracy.
    - The model is AASIST pretrained on ASVspoof LA data; its accuracy on
      other distributions is unknown without proper evaluation.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ── Path setup ─────────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).parent.resolve()
_ROOT = _TESTS_DIR.parent.resolve()
_BACKEND_DIR = _ROOT / "backend"
_ML_DIR = _ROOT / "ml"

for _p in [str(_BACKEND_DIR), str(_ML_DIR), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Data directories ───────────────────────────────────────────────────────
REAL_DIR = _ROOT / "data" / "real"
FAKE_DIR = _ROOT / "data" / "fake"


def find_wav_files(directory: Path) -> List[Path]:
    """Return sorted list of .wav files in directory (non-recursive)."""
    return sorted(directory.glob("*.wav")) + sorted(directory.glob("*.wave"))


def print_header() -> None:
    print("\n" + "=" * 56)
    print("  VoiceGuard — Audio Evaluation (Phase 2)")
    print("=" * 56)
    print(f"  Real samples dir : {REAL_DIR}")
    print(f"  Fake samples dir : {FAKE_DIR}")


def print_section(title: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def run_file(analyzer: Any, wav_path: Path, expected: str) -> Dict[str, Any]:
    """Run analyzer on one file and return a result dict."""
    t_start = time.perf_counter()
    result = analyzer.analyze(str(wav_path))
    t_wall = round(time.perf_counter() - t_start, 3)

    return {
        "filename": wav_path.name,
        "expected": expected,
        "prediction": result.prediction,
        "spoof_probability": result.spoof_probability,
        "chunks_analyzed": result.chunks_analyzed,
        "audio_duration_s": result.audio_duration_s,
        "inference_time_s": result.total_inference_time_s,
        "wall_time_s": t_wall,
        "error": result.error,
        "correct": (result.prediction == expected) if not result.error else None,
    }


def print_file_result(r: Dict[str, Any]) -> None:
    print(f"\n  File     : {r['filename']}")
    if r["error"]:
        print(f"  ❌ ERROR : {r['error']}")
        return
    verdict = "✅" if r["correct"] else "❌"
    print(f"  Expected : {r['expected'].upper()}")
    print(f"  Predicted: {r['prediction'].upper()}  {verdict}")
    print(f"  Spoof P  : {r['spoof_probability']:.4f}  ({r['spoof_probability']*100:.2f}%)")
    print(f"  Chunks   : {r['chunks_analyzed']}")
    print(f"  Duration : {r['audio_duration_s']:.2f}s")
    print(f"  Inference: {r['inference_time_s']:.3f}s  (wall: {r['wall_time_s']:.3f}s)")


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute binary classification metrics.
    Positive class = "spoof", Negative class = "real".
    Only includes results that did not error.
    """
    valid = [r for r in results if r["error"] is None]
    if len(valid) < 2:
        return {}

    # Count classes
    n_real = sum(1 for r in valid if r["expected"] == "real")
    n_spoof = sum(1 for r in valid if r["expected"] == "spoof")

    if n_real == 0 or n_spoof == 0:
        return {}  # Need at least one sample per class

    tp = sum(1 for r in valid if r["expected"] == "spoof" and r["prediction"] == "spoof")
    tn = sum(1 for r in valid if r["expected"] == "real"  and r["prediction"] == "real")
    fp = sum(1 for r in valid if r["expected"] == "real"  and r["prediction"] == "spoof")
    fn = sum(1 for r in valid if r["expected"] == "spoof" and r["prediction"] == "real")

    total = len(valid)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "total": total, "n_real": n_real, "n_spoof": n_spoof,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def print_metrics(m: Dict[str, Any]) -> None:
    print(f"\n{'=' * 56}")
    print("  Evaluation Metrics")
    print(f"{'=' * 56}")
    print(f"  Total samples   : {m['total']}  ({m['n_real']} real, {m['n_spoof']} spoof)")
    print(f"\n  Confusion Matrix (positive = spoof):")
    print(f"    TP (spoof→spoof): {m['tp']}    FN (spoof→real): {m['fn']}")
    print(f"    FP (real→spoof) : {m['fp']}    TN (real→real) : {m['tn']}")
    print(f"\n  Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-score  : {m['f1']:.4f}")
    print()
    print("  ⚠️  These metrics reflect YOUR provided samples only.")
    print("  ⚠️  A small set does NOT prove model accuracy.")
    print("  ⚠️  AASIST was trained on ASVspoof LA; performance on")
    print("  ⚠️  other distributions is not guaranteed.")


def main() -> None:
    print_header()

    real_files = find_wav_files(REAL_DIR)
    fake_files = find_wav_files(FAKE_DIR)

    if not real_files and not fake_files:
        print("\n  ⚠️  No WAV files found in data/real/ or data/fake/")
        print("  Place .wav files there and re-run this script.")
        print("  See README.md → Testing section.\n")
        sys.exit(0)

    # ── Load analyzer (loads AASIST model once) ────────────────────────────
    print("\n  Loading AASIST model (may download ~17 MB on first run)…")
    try:
        from services.analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        print("  Model ready.\n")
    except Exception as exc:
        print(f"\n  ❌ Failed to load AASIST model:\n  {exc}")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []

    # ── Real audio ─────────────────────────────────────────────────────────
    if real_files:
        print_section(f"REAL AUDIO  ({len(real_files)} file(s))")
        for wav in real_files:
            r = run_file(analyzer, wav, "real")
            all_results.append(r)
            print_file_result(r)
    else:
        print("\n  (No real audio files found in data/real/)")

    # ── Fake audio ─────────────────────────────────────────────────────────
    if fake_files:
        print_section(f"FAKE AUDIO  ({len(fake_files)} file(s))")
        for wav in fake_files:
            r = run_file(analyzer, wav, "spoof")
            all_results.append(r)
            print_file_result(r)
    else:
        print("\n  (No fake audio files found in data/fake/)")

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_results)
    if metrics:
        print_metrics(metrics)
    else:
        valid = [r for r in all_results if not r["error"]]
        print(f"\n{'=' * 56}")
        print("  Metrics not computed.")
        if len(valid) < 2:
            print("  Reason: Need at least 2 files without errors.")
        else:
            print("  Reason: Need at least 1 real and 1 fake sample.")
        print(f"{'=' * 56}\n")


if __name__ == "__main__":
    main()
