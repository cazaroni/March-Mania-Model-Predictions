"""End-to-end reproducible NCAA modeling pipeline (phases 0-5)."""

from __future__ import annotations

import os
from pathlib import Path
import time
import traceback

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


def _output_suffix() -> str:
    suffix = os.environ.get("NCAA_OUTPUT_SUFFIX", "").strip()
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def _oof_csv(stem: str) -> Path:
    return OOF_DIR / f"{stem}{_output_suffix()}.csv"


def _eval_csv(stem: str) -> Path:
    return EVAL_DIR / f"{stem}{_output_suffix()}.csv"


def _submission_csv(stem: str) -> Path:
    return SUBMISSIONS_DIR / f"{stem}{_output_suffix()}.csv"


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


def _build_graph_games_with_margin(
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    *,
    include_tourney_edges: bool = False,
) -> pd.DataFrame:
    """Build canonical matchup rows with signed margins for graph SSL tasks.

    By default uses regular-season edges only to avoid leaking tournament outcomes.
    """

    def _canonical(df: pd.DataFrame) -> pd.DataFrame:
        keep_cols = ["Season", "WTeamID", "LTeamID", "WScore", "LScore"]
        if "WLoc" in df.columns:
            keep_cols.append("WLoc")
        out = df[keep_cols].copy()
        out["Team1"] = out[["WTeamID", "LTeamID"]].min(axis=1)
        out["Team2"] = out[["WTeamID", "LTeamID"]].max(axis=1)
        out["y_true"] = (out["WTeamID"] == out["Team1"]).astype(int)
        raw_margin = out["WScore"].astype(float) - out["LScore"].astype(float)
        out["Margin"] = np.where(out["y_true"] == 1, raw_margin, -raw_margin)
        if "WLoc" not in out.columns:
            out["WLoc"] = "N"
        out["WLoc"] = out["WLoc"].fillna("N").astype(str).str.upper()
        out.loc[~out["WLoc"].isin(["H", "A", "N"]), "WLoc"] = "N"
        return out[["Season", "Team1", "Team2", "y_true", "Margin", "WLoc"]]

    reg = _canonical(reg_compact)
    reg["IsTourney"] = 0
    if include_tourney_edges:
        trn = _canonical(tourney_compact)
        trn["IsTourney"] = 1
        return pd.concat([reg, trn], ignore_index=True)
    return reg


def _encode_archetype_pair(graph_features: pd.DataFrame) -> pd.DataFrame:
    if graph_features.empty or "ArchetypePair" not in graph_features.columns:
        return graph_features

    out = graph_features.copy()
    out["ArchetypePair_enc"] = (
        out.groupby("Season")["ArchetypePair"]
        .transform(lambda s: pd.factorize(s, sort=True)[0])
        .astype(np.int32)
    )
    out = out.drop(columns=["ArchetypePair"])
    return out


def _validate_lstm_conference_structure(
    embeddings: dict[tuple[int, int], np.ndarray],
    team_conf: pd.DataFrame,
    gender: str,
) -> None:
    if not embeddings or team_conf.empty:
        print(f"[PHASE7] LSTM validation skipped for gender={gender.upper()} (missing embeddings or conferences)", flush=True)
        return

    seasons = sorted({k[0] for k in embeddings.keys()})
    if not seasons:
        print(f"[PHASE7] LSTM validation skipped for gender={gender.upper()} (no seasons)", flush=True)
        return
    season = int(seasons[-1])

    conf_cols = [c for c in ["Season", "TeamID", "ConfAbbrev", "DayNum"] if c in team_conf.columns]
    conf = team_conf[conf_cols].copy()
    if "ConfAbbrev" not in conf.columns:
        print(f"[PHASE7] LSTM validation skipped for gender={gender.upper()} (ConfAbbrev missing)", flush=True)
        return

    if "DayNum" in conf.columns:
        conf = conf.sort_values(["Season", "TeamID", "DayNum"]).drop_duplicates(["Season", "TeamID"], keep="last")
    else:
        conf = conf.drop_duplicates(["Season", "TeamID"], keep="last")
    conf = conf[conf["Season"].astype(int) == season]

    rows: list[tuple[int, str, np.ndarray]] = []
    for row in conf.itertuples(index=False):
        key = (int(row.Season), int(row.TeamID))
        z = embeddings.get(key)
        if z is None:
            continue
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 1 or z.size == 0:
            continue
        if float(np.linalg.norm(z)) <= 0.0:
            continue
        rows.append((int(row.TeamID), str(row.ConfAbbrev), z))

    if len(rows) < 4:
        print(f"[PHASE7] LSTM validation skipped for gender={gender.upper()} (insufficient teams)", flush=True)
        return

    confs = np.asarray([r[1] for r in rows])
    z = np.vstack([r[2] for r in rows]).astype(np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(norms, 1e-12, None)
    cos = z @ z.T

    iu = np.triu_indices(len(rows), k=1)
    pair_cos = cos[iu]
    same_mask = confs[iu[0]] == confs[iu[1]]
    cross_mask = ~same_mask

    same_mean = float(pair_cos[same_mask].mean()) if np.any(same_mask) else float("nan")
    cross_mean = float(pair_cos[cross_mask].mean()) if np.any(cross_mask) else float("nan")

    print(
        f"[PHASE7] LSTM validation - same-conf similarity: {same_mean:.4f}, cross-conf similarity: {cross_mean:.4f}",
        flush=True,
    )
    if np.isfinite(same_mean) and np.isfinite(cross_mean) and not (same_mean > cross_mean):
        print("[PHASE7] WARNING: same-conf not higher than cross-conf - embeddings may lack structure", flush=True)


def _run_gender_pipeline(
    *,
    gender: str,
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    team_features: pd.DataFrame,
    include_extra_models: bool = False,
    graph_features: pd.DataFrame | None = None,
    temporal_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = build_game_training_rows(reg_compact, tourney_compact)
    train_df = build_matchup_matrix(games, team_features)
    
    # Phase 6: Integrate graph features if available
    if graph_features is not None and not graph_features.empty:
        merge_keys = ["Season", "Team1", "Team2"]
        gf = graph_features.copy()
        feature_cols = [c for c in gf.columns if c not in merge_keys]
        if gf.duplicated(subset=merge_keys).any():
            numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(gf[c])]
            gf = gf.groupby(merge_keys, as_index=False)[numeric_cols].mean()

        pre_rows = len(train_df)
        train_df = train_df.merge(gf, on=merge_keys, how="left")
        post_rows = len(train_df)
        if post_rows != pre_rows:
            raise RuntimeError(
                f"Graph feature merge changed row count for gender={gender}: {pre_rows} -> {post_rows}."
            )

        if "GraphWinProb" in train_df.columns:
            train_df["GraphWinProb"] = train_df["GraphWinProb"].fillna(0.5)
            use_graph_winprob = os.environ.get("NCAA_PHASE6_USE_GRAPH_WINPROB", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if not use_graph_winprob:
                train_df = train_df.drop(columns=["GraphWinProb"])
        graph_cols = [
            c
            for c in train_df.columns
            if c.startswith(("Embed", "Cluster", "Neighbor", "Graph", "Archetype"))
        ]
        print(f"[PIPELINE] gender={gender.upper()} integrated {len(graph_cols)} graph features", flush=True)

    # Phase 7: Integrate temporal features if available
    if temporal_features is not None and not temporal_features.empty:
        merge_keys = ["Season", "Team1", "Team2"]
        tf = temporal_features.copy()
        temporal_cols = [c for c in tf.columns if c.startswith(("LSTM_", "GRU_"))]

        if tf.duplicated(subset=merge_keys).any():
            numeric_cols = [c for c in temporal_cols if pd.api.types.is_numeric_dtype(tf[c])]
            tf = tf.groupby(merge_keys, as_index=False)[numeric_cols].mean()

        pre_rows = len(train_df)
        train_df = train_df.merge(tf, on=merge_keys, how="left")
        post_rows = len(train_df)
        if post_rows != pre_rows:
            raise RuntimeError(
                f"Temporal feature merge changed row count for gender={gender}: {pre_rows} -> {post_rows}."
            )

        temporal_cols = [c for c in train_df.columns if c.startswith(("LSTM_", "GRU_"))]
        if temporal_cols:
            train_df[temporal_cols] = train_df[temporal_cols].fillna(0.0)
        print(f"[PIPELINE] gender={gender.upper()} integrated {len(temporal_cols)} temporal features", flush=True)
    
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
            base_df[out_cols].to_csv(_oof_csv(f"oof_{str(base_name).lower()}_{gender.lower()}"), index=False)

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
            stack_out.to_csv(_oof_csv(f"oof_stack_{gender.lower()}"), index=False)
            if not stack_diag.empty:
                stack_diag.to_csv(_eval_csv(f"stack_diagnostics_{gender.lower()}"), index=False)

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
            stack_t_out.to_csv(_oof_csv(f"oof_stack_tourney_{gender.lower()}"), index=False)
            if not stack_t_diag.empty:
                stack_t_diag.to_csv(_eval_csv(f"stack_diagnostics_tourney_{gender.lower()}"), index=False)

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
    out.to_csv(_submission_csv("submission_template"), index=False)


def _run_calibration_phase(
    *,
    gender: str,
    fold_metrics: pd.DataFrame,
    bins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 5: rolling calibration over selected model OOF predictions."""
    methods = [m.strip() for m in os.environ.get("NCAA_CAL_METHODS", "platt,temperature").split(",") if m.strip()]
    scopes = [s.strip() for s in os.environ.get("NCAA_CAL_SCOPES", "all,tournament_only").split(",") if s.strip()]
    shrink_values = [
        float(x.strip())
        for x in os.environ.get("NCAA_CAL_SHRINKS", "0.0,0.02,0.04,0.06").split(",")
        if x.strip()
    ]

    candidate_files = {
        "logreg": _oof_csv(f"oof_logreg_{gender.lower()}"),
        "stack": _oof_csv(f"oof_stack_{gender.lower()}"),
        "stack_tourney": _oof_csv(f"oof_stack_tourney_{gender.lower()}"),
    }
    best_rows: list[dict[str, float | str]] = []

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
            best_path = _oof_csv(f"oof_{base_name}_cal_best_{gender.lower()}")
            best_oof[[c for c in ["Season", "Team1", "Team2", "y_true", "pred", "IsTourney"] if c in best_oof.columns]].to_csv(
                best_path,
                index=False,
            )
            print(f"[CAL] gender={gender.upper()} source={base_name} best={best_label} brier={best_tourney_brier:.6f}", flush=True)
            best_rows.append(
                {
                    "gender": gender.lower(),
                    "source_model": base_name,
                    "best_model_label": best_label,
                    "best_tournament_brier": float(best_tourney_brier),
                    "best_oof_path": str(best_path),
                }
            )

    if best_rows:
        pd.DataFrame(best_rows).sort_values("best_tournament_brier").to_csv(
            _eval_csv(f"calibration_best_{gender.lower()}"),
            index=False,
        )

    return fold_metrics, bins


def main() -> None:
    _ensure_dirs()
    data = load_data(DATA_DIR)
    include_extra_models = os.environ.get("NCAA_ENABLE_EXTRA_MODELS", "0").strip().lower() in {"1", "true", "yes", "y"}
    phase6_enabled = os.environ.get("NCAA_PHASE6_ENABLE", "1").strip().lower() in {"1", "true", "yes", "y"}

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

    # Phase 6: Graph embeddings + feature extraction
    m_graph_features = pd.DataFrame()
    w_graph_features = pd.DataFrame()
    if phase6_enabled:
        print("[PHASE6] starting graph embedding phase...", flush=True)
        try:
            from graph_embed import _load_embeddings, extract_graph_features, train_graph_embedding

            include_tourney_edges = os.environ.get("NCAA_PHASE6_INCLUDE_TOURNEY_EDGES", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }

            m_games = _build_graph_games_with_margin(
                data["m_reg"],
                data["m_tourney"],
                include_tourney_edges=include_tourney_edges,
            )
            w_games = _build_graph_games_with_margin(
                data["w_reg"],
                data["w_tourney"],
                include_tourney_edges=include_tourney_edges,
            )

            phase6_dim = int(os.environ.get("NCAA_PHASE6_EMBEDDING_DIM", "64"))
            phase6_layers = int(os.environ.get("NCAA_PHASE6_GNN_LAYERS", "2"))
            phase6_epochs = int(os.environ.get("NCAA_PHASE6_EPOCHS", "30"))
            phase6_lr = float(os.environ.get("NCAA_PHASE6_LR", "0.001"))
            retrain = os.environ.get("NCAA_PHASE6_RETRAIN", "0").strip().lower() in {"1", "true", "yes", "y"}

            # On ROCm builds (MI210), torch.cuda.is_available() is also the expected check.
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

            m_emb_path = FEATURES_DIR / "graph_embeddings_m.parquet"
            w_emb_path = FEATURES_DIR / "graph_embeddings_w.parquet"

            if not retrain and m_emb_path.exists():
                print(f"[PHASE6] loading cached embeddings from {m_emb_path}", flush=True)
                m_emb_dict = _load_embeddings(m_emb_path)
                m_graph_winprob = pd.DataFrame()
            else:
                m_emb_dict, _, m_graph_winprob = train_graph_embedding(
                    games=m_games,
                    team_features=m_team_features,
                    game_locations=m_games[["Season", "Team1", "Team2", "WLoc"]],
                    gender="m",
                    embedding_dim=phase6_dim,
                    num_layers=phase6_layers,
                    epochs=phase6_epochs,
                    lr=phase6_lr,
                    device=device,
                    embeddings_path=m_emb_path,
                )
            m_graph_features = extract_graph_features(
                games=m_games,
                team_features=m_team_features,
                embeddings_dict=m_emb_dict,
                gender="m",
                graph_winprob=m_graph_winprob,
            )
            m_graph_features = _encode_archetype_pair(m_graph_features)

            if not retrain and w_emb_path.exists():
                print(f"[PHASE6] loading cached embeddings from {w_emb_path}", flush=True)
                w_emb_dict = _load_embeddings(w_emb_path)
                w_graph_winprob = pd.DataFrame()
            else:
                w_emb_dict, _, w_graph_winprob = train_graph_embedding(
                    games=w_games,
                    team_features=w_team_features,
                    game_locations=w_games[["Season", "Team1", "Team2", "WLoc"]],
                    gender="w",
                    embedding_dim=phase6_dim,
                    num_layers=phase6_layers,
                    epochs=phase6_epochs,
                    lr=phase6_lr,
                    device=device,
                    embeddings_path=w_emb_path,
                )
            w_graph_features = extract_graph_features(
                games=w_games,
                team_features=w_team_features,
                embeddings_dict=w_emb_dict,
                gender="w",
                graph_winprob=w_graph_winprob,
            )
            w_graph_features = _encode_archetype_pair(w_graph_features)

            if not m_graph_features.empty:
                m_graph_features.to_csv(FEATURES_DIR / "graph_features_m.csv", index=False)
            if not w_graph_features.empty:
                w_graph_features.to_csv(FEATURES_DIR / "graph_features_w.csv", index=False)

            print("[PHASE6] graph embedding completed successfully", flush=True)
        except Exception as e:
            print(f"[PHASE6] WARNING: graph embedding failed ({type(e).__name__}: {e}), continuing without graph features", flush=True)
            traceback.print_exc()
            m_graph_features = pd.DataFrame()
            w_graph_features = pd.DataFrame()
    else:
        print("[PHASE6] skipped (NCAA_PHASE6_ENABLE=0)", flush=True)

    # Phase 7: Temporal embeddings (LSTM + GRU)
    print("[PHASE7] starting temporal embedding phase...", flush=True)
    m_temporal_features = pd.DataFrame()
    w_temporal_features = pd.DataFrame()
    try:
        from temporal_embed import build_game_sequences, extract_temporal_features, train_temporal_model

        phase7_enabled = os.environ.get("NCAA_PHASE7_ENABLE", "1").strip().lower() in {"1", "true", "yes", "y"}
        if not phase7_enabled:
            print("[PHASE7] disabled via NCAA_PHASE7_ENABLE=0", flush=True)
            raise ValueError("phase7 disabled")

        phase7_retrain = os.environ.get("NCAA_PHASE7_RETRAIN", "1").strip().lower() in {"1", "true", "yes", "y"}
        phase7_hidden = int(os.environ.get("NCAA_PHASE7_HIDDEN_DIM", "64"))
        phase7_lstm_epochs = int(os.environ.get("NCAA_PHASE7_LSTM_EPOCHS", "20"))
        phase7_gru_epochs = int(os.environ.get("NCAA_PHASE7_GRU_EPOCHS", "20"))
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

        m_temporal_path = FEATURES_DIR / "temporal_features_m.csv"
        w_temporal_path = FEATURES_DIR / "temporal_features_w.csv"

        if (not phase7_retrain) and m_temporal_path.exists():
            print(f"[PHASE7] loading cached temporal features from {m_temporal_path}", flush=True)
            m_temporal_features = pd.read_csv(m_temporal_path)
        else:
            m_sequences = build_game_sequences(data["m_reg_det"], gender="m")
            m_lstm_emb = train_temporal_model(
                sequences=m_sequences,
                games=data["m_reg_det"],
                model_type="lstm",
                hidden_dim=phase7_hidden,
                epochs=phase7_lstm_epochs,
                lr=1e-3,
                device=device,
            )
            m_gru_emb = train_temporal_model(
                sequences=m_sequences,
                games=data["m_reg_det"],
                model_type="gru",
                hidden_dim=phase7_hidden,
                epochs=phase7_gru_epochs,
                lr=1e-3,
                device=device,
            )
            _validate_lstm_conference_structure(m_lstm_emb, data["m_team_conf"], gender="m")
            m_games_all = build_game_training_rows(data["m_reg"], data["m_tourney"])
            m_temporal_features = extract_temporal_features(m_games_all, m_lstm_emb, m_gru_emb)

        if (not phase7_retrain) and w_temporal_path.exists():
            print(f"[PHASE7] loading cached temporal features from {w_temporal_path}", flush=True)
            w_temporal_features = pd.read_csv(w_temporal_path)
        else:
            w_sequences = build_game_sequences(data["w_reg_det"], gender="w")
            w_lstm_emb = train_temporal_model(
                sequences=w_sequences,
                games=data["w_reg_det"],
                model_type="lstm",
                hidden_dim=phase7_hidden,
                epochs=phase7_lstm_epochs,
                lr=1e-3,
                device=device,
            )
            w_gru_emb = train_temporal_model(
                sequences=w_sequences,
                games=data["w_reg_det"],
                model_type="gru",
                hidden_dim=phase7_hidden,
                epochs=phase7_gru_epochs,
                lr=1e-3,
                device=device,
            )
            _validate_lstm_conference_structure(w_lstm_emb, data["w_team_conf"], gender="w")
            w_games_all = build_game_training_rows(data["w_reg"], data["w_tourney"])
            w_temporal_features = extract_temporal_features(w_games_all, w_lstm_emb, w_gru_emb)

        m_temporal_features.to_csv(m_temporal_path, index=False)
        w_temporal_features.to_csv(w_temporal_path, index=False)
    except Exception as e:
        print(f"[PHASE7] WARNING: temporal embedding failed ({type(e).__name__}: {e}), continuing without", flush=True)
        traceback.print_exc()
        m_temporal_features = pd.DataFrame()
        w_temporal_features = pd.DataFrame()
        pd.DataFrame().to_csv(FEATURES_DIR / "temporal_features_m.csv", index=False)
        pd.DataFrame().to_csv(FEATURES_DIR / "temporal_features_w.csv", index=False)

    # Rolling CV + base models + stacking
    m_oof, m_fold_metrics, m_bins = _run_gender_pipeline(
        gender="m",
        reg_compact=data["m_reg"],
        tourney_compact=data["m_tourney"],
        team_features=m_team_features,
        include_extra_models=include_extra_models,
        graph_features=m_graph_features,
        temporal_features=m_temporal_features,
    )
    w_oof, w_fold_metrics, w_bins = _run_gender_pipeline(
        gender="w",
        reg_compact=data["w_reg"],
        tourney_compact=data["w_tourney"],
        team_features=w_team_features,
        include_extra_models=include_extra_models,
        graph_features=w_graph_features,
        temporal_features=w_temporal_features,
    )

    # Save evaluation outputs
    m_fold_metrics, m_bins = _run_calibration_phase(gender="m", fold_metrics=m_fold_metrics, bins=m_bins)
    w_fold_metrics, w_bins = _run_calibration_phase(gender="w", fold_metrics=w_fold_metrics, bins=w_bins)

    # Save evaluation outputs
    if not m_fold_metrics.empty:
        m_fold_metrics.to_csv(_eval_csv("fold_metrics_m"), index=False)
    if not w_fold_metrics.empty:
        w_fold_metrics.to_csv(_eval_csv("fold_metrics_w"), index=False)
    if not m_bins.empty:
        m_bins.to_csv(_eval_csv("calibration_bins_m"), index=False)
    if not w_bins.empty:
        w_bins.to_csv(_eval_csv("calibration_bins_w"), index=False)

    _write_submission_template(data["sub_stage2"])

    # Console summary
    print("\n=== Pipeline complete ===")
    print(f"Men OOF rows: {len(m_oof):,}")
    print(f"Women OOF rows: {len(w_oof):,}")
    print(f"OOF directory: {OOF_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Submission template: {_submission_csv('submission_template')}")

    if not m_fold_metrics.empty:
        print("\nMen metrics (mean by model/regime):")
        print(m_fold_metrics.groupby(["model", "regime"], as_index=False)[["brier", "log_loss"]].mean().sort_values("brier"))
    if not w_fold_metrics.empty:
        print("\nWomen metrics (mean by model/regime):")
        print(w_fold_metrics.groupby(["model", "regime"], as_index=False)[["brier", "log_loss"]].mean().sort_values("brier"))


if __name__ == "__main__":
    main()
