"""End-to-end reproducible NCAA modeling pipeline (phases 0-4)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from cv import run_rolling_cv
from eval import evaluate_by_regime
from features_baseline import build_basic_team_features, load_data
from matchups import build_game_training_rows, build_matchup_matrix, feature_columns_for_training
from models import build_base_model_factories
from ratings import build_and_save_ratings
from stack import build_stack_features, rolling_stack_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
OOF_DIR = ROOT / "oof"
SUBMISSIONS_DIR = ROOT / "submissions"
EVAL_DIR = ROOT / "eval"


def _ensure_dirs() -> None:
    for d in [FEATURES_DIR, OOF_DIR, SUBMISSIONS_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _combine_team_features(basic: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    out = basic.merge(ratings, on=["Season", "TeamID"], how="left")
    numeric_cols = [c for c in out.columns if c not in ["Season", "TeamID"]]
    out[numeric_cols] = out.groupby("Season")[numeric_cols].transform(lambda s: s.fillna(s.median()))
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def _run_gender_pipeline(
    *,
    gender: str,
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = build_game_training_rows(reg_compact, tourney_compact)
    train_df = build_matchup_matrix(games, team_features)
    feature_cols = feature_columns_for_training(train_df)

    factories = build_base_model_factories()
    base_oof_parts = []
    fold_metrics_parts = []
    bin_parts = []

    for model_name, factory in factories.items():
        cv_res = run_rolling_cv(
            train_df,
            feature_cols=feature_cols,
            model_name=model_name,
            model_factory=factory,
            season_col="Season",
            y_col="y_true",
            meta_cols=["Season", "Team1", "Team2", "IsTourney"],
        )
        if cv_res.oof.empty:
            continue

        base_oof_parts.append(cv_res.oof)
        fold_metrics_parts.append(cv_res.fold_metrics)
        if not cv_res.calibration_bins.empty:
            bin_parts.append(cv_res.calibration_bins)

        if model_name in {"logreg", "hgb"}:
            baseline_path = OOF_DIR / f"oof_{model_name}_{gender.lower()}.csv"
            cv_res.oof[["Season", "Team1", "Team2", "y_true", "pred", "model"]].to_csv(baseline_path, index=False)

    base_oof = pd.concat(base_oof_parts, ignore_index=True) if base_oof_parts else pd.DataFrame()
    fold_metrics = pd.concat(fold_metrics_parts, ignore_index=True) if fold_metrics_parts else pd.DataFrame()
    bins = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()

    if not base_oof.empty:
        stack_wide, stack_features = build_stack_features(base_oof)
        stack_oof = rolling_stack_oof(stack_wide, feature_cols=stack_features)
        if not stack_oof.empty:
            stack_out = stack_oof[["Season", "Team1", "Team2", "y_true", "pred"]].copy()
            stack_out.to_csv(OOF_DIR / f"oof_stack_{gender.lower()}.csv", index=False)

            stack_eval_metrics, stack_eval_bins = evaluate_by_regime(stack_oof, y_col="y_true", pred_col="pred", tourney_flag_col="IsTourney")
            stack_eval_metrics["model"] = "stack"
            stack_eval_metrics["valid_season"] = -1
            fold_metrics = pd.concat([fold_metrics, stack_eval_metrics], ignore_index=True)
            if not stack_eval_bins.empty:
                stack_eval_bins["model"] = "stack"
                stack_eval_bins["valid_season"] = -1
                bins = pd.concat([bins, stack_eval_bins], ignore_index=True)

    return base_oof, fold_metrics, bins


def _write_submission_template(sub_df: pd.DataFrame) -> None:
    out = sub_df[["ID"]].copy()
    out["Pred"] = 0.5
    out.to_csv(SUBMISSIONS_DIR / "submission_template.csv", index=False)


def main() -> None:
    _ensure_dirs()
    data = load_data(DATA_DIR)

    # Ratings layer (Phase 2)
    m_ratings = build_and_save_ratings(
        DATA_DIR,
        FEATURES_DIR,
        reg_compact_file="MRegularSeasonCompactResults.csv",
        reg_detailed_file="MRegularSeasonDetailedResults.csv",
        team_conf_file="MTeamConferences.csv",
        output_file="team_ratings_m.parquet",
        massey_file="MMasseyOrdinals.csv",
    )
    w_ratings = build_and_save_ratings(
        DATA_DIR,
        FEATURES_DIR,
        reg_compact_file="WRegularSeasonCompactResults.csv",
        reg_detailed_file="WRegularSeasonDetailedResults.csv",
        team_conf_file="WTeamConferences.csv",
        output_file="team_ratings_w.parquet",
        massey_file=None,
    )

    # Baseline team features + ratings
    m_basic = build_basic_team_features(data["m_reg"])
    w_basic = build_basic_team_features(data["w_reg"])
    m_team_features = _combine_team_features(m_basic, m_ratings)
    w_team_features = _combine_team_features(w_basic, w_ratings)

    # Rolling CV + base models + stacking
    m_oof, m_fold_metrics, m_bins = _run_gender_pipeline(
        gender="m",
        reg_compact=data["m_reg"],
        tourney_compact=data["m_tourney"],
        team_features=m_team_features,
    )
    w_oof, w_fold_metrics, w_bins = _run_gender_pipeline(
        gender="w",
        reg_compact=data["w_reg"],
        tourney_compact=data["w_tourney"],
        team_features=w_team_features,
    )

    # Save evaluation outputs
    if not m_fold_metrics.empty:
        m_fold_metrics.to_csv(EVAL_DIR / "fold_metrics_m.csv", index=False)
    if not w_fold_metrics.empty:
        w_fold_metrics.to_csv(EVAL_DIR / "fold_metrics_w.csv", index=False)
    if not m_bins.empty:
        m_bins.to_csv(EVAL_DIR / "calibration_bins_m.csv", index=False)
    if not w_bins.empty:
        w_bins.to_csv(EVAL_DIR / "calibration_bins_w.csv", index=False)

    _write_submission_template(data["sub_stage2"])

    # Console summary
    print("\n=== Pipeline complete ===")
    print(f"Men OOF rows: {len(m_oof):,}")
    print(f"Women OOF rows: {len(w_oof):,}")
    print(f"OOF directory: {OOF_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Submission template: {SUBMISSIONS_DIR / 'submission_template.csv'}")

    if not m_fold_metrics.empty:
        print("\nMen metrics (mean by model/regime):")
        print(m_fold_metrics.groupby(["model", "regime"], as_index=False)[["brier", "log_loss"]].mean().sort_values("brier"))
    if not w_fold_metrics.empty:
        print("\nWomen metrics (mean by model/regime):")
        print(w_fold_metrics.groupby(["model", "regime"], as_index=False)[["brier", "log_loss"]].mean().sort_values("brier"))


if __name__ == "__main__":
    main()
