"""End-to-end reproducible NCAA modeling pipeline (phases 0-4)."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cv import run_rolling_cv
from calibration import rolling_calibrate_oof
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

    def _fill_group_median(series: pd.Series) -> pd.Series:
        non_na = series.dropna()
        if non_na.empty:
            return series.fillna(0.0)
        return series.fillna(float(non_na.median()))

    out[numeric_cols] = out.groupby("Season")[numeric_cols].transform(_fill_group_median)
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def _run_gender_pipeline(
    *,
    gender: str,
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    team_features: pd.DataFrame,
    include_extra_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = build_game_training_rows(reg_compact, tourney_compact)
    train_df = build_matchup_matrix(games, team_features)
    # Preserve one-row-per-game identity for stack pivoting.
    train_df["GameRowID"] = np.arange(len(train_df), dtype=np.int64)
    feature_cols = feature_columns_for_training(train_df)
    prior_meta_candidates = [
        "Diff_Elo",
        "Diff_BT",
        "Diff_AdjNet",
        "Diff_AdjORtg",
        "Diff_AdjDRtg",
        "Diff_PythagWinPct",
        "Diff_LuckResidual",
    ]
    prior_meta_cols = [c for c in prior_meta_candidates if c in train_df.columns]

    factories = build_base_model_factories(include_extras=include_extra_models)
    base_oof_parts = []
    fold_metrics_parts = []
    bin_parts = []

    for model_name, factory in factories.items():
        model_start = time.perf_counter()
        print(f"[PIPELINE] gender={gender.upper()} model={model_name} starting", flush=True)
        cv_res = run_rolling_cv(
            train_df,
            feature_cols=feature_cols,
            model_name=model_name,
            model_factory=factory,
            season_col="Season",
            y_col="y_true",
            meta_cols=["Season", "GameRowID", "Team1", "Team2", "IsTourney", *prior_meta_cols],
            progress_label=f"{gender.upper()}:{model_name}",
            progress=True,
        )
        print(
            f"[PIPELINE] gender={gender.upper()} model={model_name} finished in {(time.perf_counter() - model_start)/60.0:.2f}m",
            flush=True,
        )
        if cv_res.oof.empty:
            continue

        base_oof_parts.append(cv_res.oof)
        fold_metrics_parts.append(cv_res.fold_metrics)
        if not cv_res.calibration_bins.empty:
            bin_parts.append(cv_res.calibration_bins)

    base_oof = pd.concat(base_oof_parts, ignore_index=True) if base_oof_parts else pd.DataFrame()
    fold_metrics = pd.concat(fold_metrics_parts, ignore_index=True) if fold_metrics_parts else pd.DataFrame()
    bins = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()

    if not base_oof.empty:
        for base_name, base_df in base_oof.groupby("model", sort=True):
            out_cols = ["Season", "GameRowID", "Team1", "Team2", "y_true", "pred", "model"]
            if "IsTourney" in base_df.columns:
                out_cols.insert(5, "IsTourney")
            out_cols.extend([c for c in prior_meta_cols if c in base_df.columns])
            base_df[out_cols].to_csv(OOF_DIR / f"oof_{str(base_name).lower()}_{gender.lower()}.csv", index=False)

    if not base_oof.empty:
        stack_tourney_weight = float(os.environ.get("NCAA_STACK_TOURNEY_WEIGHT", "4.0"))
        stack_wide, stack_features = build_stack_features(base_oof)
        stack_oof, stack_diag = rolling_stack_oof(
            stack_wide,
            feature_cols=stack_features,
            tournament_weight=1.0,
            return_diagnostics=True,
        )
        if not stack_oof.empty:
            stack_out_cols = ["Season", "Team1", "Team2", "y_true", "pred"]
            if "IsTourney" in stack_oof.columns:
                stack_out_cols.insert(4, "IsTourney")
            stack_out = stack_oof[stack_out_cols].copy()
            stack_out.to_csv(OOF_DIR / f"oof_stack_{gender.lower()}.csv", index=False)
            if not stack_diag.empty:
                stack_diag.to_csv(EVAL_DIR / f"stack_diagnostics_{gender.lower()}.csv", index=False)

            stack_eval_metrics, stack_eval_bins = evaluate_by_regime(stack_oof, y_col="y_true", pred_col="pred", tourney_flag_col="IsTourney")
            stack_eval_metrics["model"] = "stack"
            stack_eval_metrics["valid_season"] = -1
            fold_metrics = pd.concat([fold_metrics, stack_eval_metrics], ignore_index=True)
            if not stack_eval_bins.empty:
                stack_eval_bins["model"] = "stack"
                stack_eval_bins["valid_season"] = -1
                bins = pd.concat([bins, stack_eval_bins], ignore_index=True)

        # Tournament-optimized stack variant.
        stack_t_oof, stack_t_diag = rolling_stack_oof(
            stack_wide,
            feature_cols=stack_features,
            tournament_weight=stack_tourney_weight,
            return_diagnostics=True,
        )
        if not stack_t_oof.empty:
            stack_t_out_cols = ["Season", "Team1", "Team2", "y_true", "pred"]
            if "IsTourney" in stack_t_oof.columns:
                stack_t_out_cols.insert(4, "IsTourney")
            stack_t_out = stack_t_oof[stack_t_out_cols].copy()
            stack_t_out.to_csv(OOF_DIR / f"oof_stack_tourney_{gender.lower()}.csv", index=False)
            if not stack_t_diag.empty:
                stack_t_diag.to_csv(EVAL_DIR / f"stack_diagnostics_tourney_{gender.lower()}.csv", index=False)

            stack_t_eval_metrics, stack_t_eval_bins = evaluate_by_regime(
                stack_t_oof,
                y_col="y_true",
                pred_col="pred",
                tourney_flag_col="IsTourney",
            )
            stack_t_eval_metrics["model"] = "stack_tourney"
            stack_t_eval_metrics["valid_season"] = -1
            fold_metrics = pd.concat([fold_metrics, stack_t_eval_metrics], ignore_index=True)
            if not stack_t_eval_bins.empty:
                stack_t_eval_bins["model"] = "stack_tourney"
                stack_t_eval_bins["valid_season"] = -1
                bins = pd.concat([bins, stack_t_eval_bins], ignore_index=True)

    return base_oof, fold_metrics, bins


def _write_submission_template(sub_df: pd.DataFrame) -> None:
    out = sub_df[["ID"]].copy()
    out["Pred"] = 0.5
    out.to_csv(SUBMISSIONS_DIR / "submission_template.csv", index=False)


def _run_calibration_phase(
    *,
    gender: str,
    fold_metrics: pd.DataFrame,
    bins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 5: rolling calibration over selected model OOF predictions."""
    methods = [m.strip() for m in os.environ.get("NCAA_CAL_METHODS", "platt,isotonic,temperature").split(",") if m.strip()]
    scopes = [s.strip() for s in os.environ.get("NCAA_CAL_SCOPES", "all,tournament_only").split(",") if s.strip()]
    shrink_values = [
        float(x.strip())
        for x in os.environ.get("NCAA_CAL_SHRINKS", "0.0,0.08").split(",")
        if x.strip()
    ]

    candidate_files = {
        "logreg": OOF_DIR / f"oof_logreg_{gender.lower()}.csv",
        "stack": OOF_DIR / f"oof_stack_{gender.lower()}.csv",
        "stack_tourney": OOF_DIR / f"oof_stack_tourney_{gender.lower()}.csv",
    }

    for base_name, path in candidate_files.items():
        if not path.exists():
            continue

        source_df = pd.read_csv(path)
        required_cols = {"Season", "y_true", "pred"}
        if not required_cols.issubset(source_df.columns):
            continue

        best_tourney_brier = float("inf")
        best_oof = pd.DataFrame()
        best_label = ""

        for method in methods:
            for scope in scopes:
                for shrink in shrink_values:
                    cal_oof = rolling_calibrate_oof(
                        source_df,
                        method=method,
                        season_col="Season",
                        y_col="y_true",
                        pred_col="pred",
                        tourney_flag_col="IsTourney",
                        fit_scope=scope,
                        shrink=shrink,
                    )
                    if cal_oof.empty:
                        continue

                    label = f"{base_name}_cal_{method}_{scope}_s{int(round(shrink * 100.0)):02d}"
                    metric_df, bins_df = evaluate_by_regime(
                        cal_oof,
                        y_col="y_true",
                        pred_col="pred",
                        tourney_flag_col="IsTourney",
                    )
                    metric_df["model"] = label
                    metric_df["valid_season"] = -1
                    fold_metrics = pd.concat([fold_metrics, metric_df], ignore_index=True)

                    if not bins_df.empty:
                        bins_df["model"] = label
                        bins_df["valid_season"] = -1
                        bins = pd.concat([bins, bins_df], ignore_index=True)

                    tourney_metric = metric_df[metric_df["regime"] == "tournament_only"]
                    if not tourney_metric.empty:
                        score = float(tourney_metric["brier"].iloc[0])
                    else:
                        score = float(metric_df[metric_df["regime"] == "all_games"]["brier"].iloc[0])

                    if score < best_tourney_brier:
                        best_tourney_brier = score
                        best_oof = cal_oof.copy()
                        best_label = label

        if not best_oof.empty:
            best_path = OOF_DIR / f"oof_{base_name}_cal_best_{gender.lower()}.csv"
            best_oof[[c for c in ["Season", "Team1", "Team2", "y_true", "pred", "IsTourney"] if c in best_oof.columns]].to_csv(
                best_path,
                index=False,
            )
            print(f"[CAL] gender={gender.upper()} source={base_name} best={best_label} brier={best_tourney_brier:.6f}", flush=True)

    return fold_metrics, bins


def main() -> None:
    _ensure_dirs()
    data = load_data(DATA_DIR)
    include_extra_models = os.environ.get("NCAA_ENABLE_EXTRA_MODELS", "0").strip().lower() in {"1", "true", "yes", "y"}

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
        include_extra_models=include_extra_models,
    )
    w_oof, w_fold_metrics, w_bins = _run_gender_pipeline(
        gender="w",
        reg_compact=data["w_reg"],
        tourney_compact=data["w_tourney"],
        team_features=w_team_features,
        include_extra_models=include_extra_models,
    )

    # Save evaluation outputs
    m_fold_metrics, m_bins = _run_calibration_phase(gender="m", fold_metrics=m_fold_metrics, bins=m_bins)
    w_fold_metrics, w_bins = _run_calibration_phase(gender="w", fold_metrics=w_fold_metrics, bins=w_bins)

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
