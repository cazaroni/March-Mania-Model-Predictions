"""Full pipeline refresh.

Run this when new Kaggle data is available.
Rebuilds ratings, features, retrains stack, recalibrates, generates submissions.

Usage:
    python run_full_pipeline.py
    python run_full_pipeline.py --skip-train
    python run_full_pipeline.py --submission-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_submission_only import generate_submission_candidates
from src.ratings import build_and_save_ratings


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
EVAL_DIR = ROOT / "eval"


def _refresh_ratings() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    build_and_save_ratings(
        DATA_DIR,
        FEATURES_DIR,
        reg_compact_file="MRegularSeasonCompactResults.csv",
        reg_detailed_file="MRegularSeasonDetailedResults.csv",
        team_conf_file="MTeamConferences.csv",
        output_file="team_ratings_m.parquet",
        massey_file="MMasseyOrdinals.csv",
    )
    build_and_save_ratings(
        DATA_DIR,
        FEATURES_DIR,
        reg_compact_file="WRegularSeasonCompactResults.csv",
        reg_detailed_file="WRegularSeasonDetailedResults.csv",
        team_conf_file="WTeamConferences.csv",
        output_file="team_ratings_w.parquet",
        massey_file=None,
    )
    print("[FULL] Ratings refresh complete.")


def _print_calibration_summary() -> None:
    for gender in ["m", "w"]:
        path = EVAL_DIR / f"calibration_best_{gender}.csv"
        if not path.exists():
            print(f"[FULL] calibration summary missing: {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"[FULL] calibration summary empty: {path}")
            continue

        sort_col = "best_tournament_brier" if "best_tournament_brier" in df.columns else None
        if sort_col is not None:
            df = df.sort_values(sort_col, ascending=True)

        best = df.iloc[0]
        model = str(best.get("source_model", "unknown"))
        label = str(best.get("best_model_label", "unknown"))
        brier = best.get("best_tournament_brier", "n/a")
        print(
            f"[FULL] Best calibration ({gender.upper()}): model={model}, "
            f"label={label}, tournament_brier={brier}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command full NCAA pipeline refresh.")
    parser.add_argument("--skip-train", action="store_true", help="Refresh artifacts but skip model retraining.")
    parser.add_argument(
        "--submission-only",
        action="store_true",
        help="Skip refresh/training and only generate submission candidates.",
    )
    parser.add_argument("--season", type=int, default=2026, help="Target submission season.")
    parser.add_argument(
        "--regime_b_scale",
        type=float,
        default=14.0,
        help="Sigmoid scale for Regime B AdjNet differential.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.submission_only:
        print("[FULL] Submission-only mode enabled; skipping ratings/features/training phases.")
    else:
        _refresh_ratings()

        if args.skip_train:
            print("[FULL] --skip-train enabled; using existing OOF/calibration artifacts.")
        else:
            # Keep imports local so submission-only mode does not require heavy deps.
            from src.train_baseline_cpu import main as run_main_training

            print("[FULL] Running main CPU training pipeline (includes calibration)...")
            run_main_training()
            print("[FULL] Training pipeline finished.")

    generated = generate_submission_candidates(
        season=int(args.season),
        m_model="stack",
        w_model="logreg",
        suffix="",
        regime_b_scale=float(args.regime_b_scale),
    )

    _print_calibration_summary()

    print("[FULL] Submission candidates generated:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
