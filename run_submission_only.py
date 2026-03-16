"""Fast submission generation without retraining.

Uses existing OOF files and feature files from last pipeline run.

Usage:
    python run_submission_only.py
    python run_submission_only.py --season 2025
    python run_submission_only.py --season 2025 --suffix v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.submission import build_submission


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
OOF_DIR = ROOT / "oof"
SUBMISSIONS_DIR = ROOT / "submissions"


def generate_submission_candidates(
    *,
    season: int,
    m_model: str,
    w_model: str,
    suffix: str,
    regime_b_scale: float,
) -> list[Path]:
    """Generate multiple submission candidates in one pass."""
    custom_suffix = suffix.strip().strip("_")
    if not custom_suffix:
        custom_suffix = f"{m_model}_{w_model}"

    candidates = [
        {"m_model": m_model, "w_model": w_model, "suffix": custom_suffix},
        {"m_model": "stack", "w_model": "logreg", "suffix": "stack_logreg"},
        {"m_model": "logreg", "w_model": "logreg", "suffix": "logreg_logreg"},
        {"m_model": "stack", "w_model": "stack", "suffix": "stack_stack"},
    ]

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in candidates:
        key = (str(c["m_model"]), str(c["w_model"]), str(c["suffix"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    outputs: list[Path] = []
    for c in deduped:
        path = build_submission(
            data_dir=DATA_DIR,
            features_dir=FEATURES_DIR,
            oof_dir=OOF_DIR,
            submissions_dir=SUBMISSIONS_DIR,
            season=int(season),
            m_model=str(c["m_model"]),
            w_model=str(c["w_model"]),
            regime_b_scale=float(regime_b_scale),
            output_suffix=str(c["suffix"]),
            clip_range=(0.05, 0.95),
        )
        outputs.append(path)
        print(f"Generated: {path}")

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast submission generation without retraining.")
    parser.add_argument("--season", type=int, default=2026, help="Target season for submission IDs.")
    parser.add_argument("--m_model", type=str, default="stack", help="Model family for men.")
    parser.add_argument("--w_model", type=str, default="logreg", help="Model family for women.")
    parser.add_argument("--suffix", type=str, default="", help="Optional output suffix.")
    parser.add_argument(
        "--regime_b_scale",
        type=float,
        default=14.0,
        help="Sigmoid scale for Regime B AdjNet differential.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_submission_candidates(
        season=int(args.season),
        m_model=str(args.m_model),
        w_model=str(args.w_model),
        suffix=str(args.suffix),
        regime_b_scale=float(args.regime_b_scale),
    )


if __name__ == "__main__":
    main()
