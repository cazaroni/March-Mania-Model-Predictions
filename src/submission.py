"""Phase 8: Tournament-aware submission generation.

Regime A - both teams are seeded (tournament-eligible):
    Use full calibrated ensemble predictions.

Regime B - one or both teams unseeded:
    Use smoothed sigmoid of AdjNet rating differential only.
    Clip conservatively to [0.05, 0.95].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from calibration import fit_calibrator
    from features_baseline import build_basic_team_features
    from matchups import build_game_training_rows, build_matchup_matrix, feature_columns_for_training
    from models import build_base_model_factories
    from stack import build_stack_features
except ImportError:
    from src.calibration import fit_calibrator
    from src.features_baseline import build_basic_team_features
    from src.matchups import build_game_training_rows, build_matchup_matrix, feature_columns_for_training
    from src.models import build_base_model_factories
    from src.stack import build_stack_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_FEATURES_DIR = ROOT / "features"
DEFAULT_OOF_DIR = ROOT / "oof"
DEFAULT_SUBMISSIONS_DIR = ROOT / "submissions"
DEFAULT_EVAL_DIR = ROOT / "eval"


@dataclass(frozen=True)
class CalibrationSpec:
    method: str
    scope: str
    shrink: float


@dataclass(frozen=True)
class SubmissionConfig:
    data_dir: Path
    features_dir: Path
    oof_dir: Path
    submissions_dir: Path
    eval_dir: Path


def load_seeds(data_dir: Path, season: int) -> set[int]:
    """Return set of TeamIDs seeded in given season, both M and W."""
    seeded: set[int] = set()
    for path in [
        data_dir / "MNCAATourneySeeds.csv",
        data_dir / "WNCAATourneySeeds.csv",
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Season" not in df.columns or "TeamID" not in df.columns:
            continue
        s = df[df["Season"].astype(int) == int(season)]
        if s.empty:
            continue
        seeded.update(s["TeamID"].astype(int).tolist())
    return seeded


def build_all_pairs(data_dir: Path, season: int) -> pd.DataFrame:
    """Build all team pairs for the target season across men and women."""
    parts: list[pd.DataFrame] = []
    file_map = {
        "M": data_dir / "MRegularSeasonCompactResults.csv",
        "W": data_dir / "WRegularSeasonCompactResults.csv",
    }
    for gender, path in file_map.items():
        if not path.exists():
            continue
        reg = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
        reg = reg[reg["Season"].astype(int) == int(season)]
        if reg.empty:
            continue
        teams = np.sort(
            pd.unique(
                pd.concat(
                    [reg["WTeamID"], reg["LTeamID"]],
                    ignore_index=True,
                )
            ).astype(int)
        )
        if len(teams) < 2:
            continue
        i, j = np.triu_indices(len(teams), k=1)
        out = pd.DataFrame(
            {
                "Season": int(season),
                "Team1": teams[i],
                "Team2": teams[j],
                "Gender": gender,
            }
        )
        out["ID"] = out["Season"].astype(str) + "_" + out["Team1"].astype(str) + "_" + out["Team2"].astype(str)
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["Season", "Team1", "Team2", "Gender", "ID"])
    return pd.concat(parts, ignore_index=True).sort_values("ID").reset_index(drop=True)


def regime_b_probability(
    pairs: pd.DataFrame,
    team_ratings: pd.DataFrame,
    scale: float = 14.0,
) -> pd.DataFrame:
    """Smooth sigmoid of AdjNet differential for non-tournament pairs."""
    out = pairs.copy()
    if out.empty:
        out["Pred"] = np.array([], dtype=float)
        return out

    ratings = team_ratings.copy()
    keep = [c for c in ["Season", "TeamID", "AdjNet"] if c in ratings.columns]
    if len(keep) < 3:
        out["Pred"] = 0.5
        return out

    ratings = ratings[keep].copy()
    t1 = ratings.rename(columns={"TeamID": "Team1", "AdjNet": "AdjNet_T1"})
    t2 = ratings.rename(columns={"TeamID": "Team2", "AdjNet": "AdjNet_T2"})

    out = out.merge(t1, on=["Season", "Team1"], how="left")
    out = out.merge(t2, on=["Season", "Team2"], how="left")

    adj1 = out["AdjNet_T1"].astype(float)
    adj2 = out["AdjNet_T2"].astype(float)
    diff = (adj1 - adj2).fillna(0.0)
    scale = max(float(scale), 1e-6)
    pred = 1.0 / (1.0 + np.exp(-diff / scale))
    out["Pred"] = np.clip(pred, 0.05, 0.95)
    return out.drop(columns=[c for c in ["AdjNet_T1", "AdjNet_T2"] if c in out.columns])


def _clip_prob(pred: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    lo, hi = float(clip_range[0]), float(clip_range[1])
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    if lo >= hi:
        lo, hi = 0.05, 0.95
    return np.clip(pred, lo, hi)


def _combine_team_features(basic: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    out = basic.merge(ratings, on=["Season", "TeamID"], how="left")
    numeric_cols = [c for c in out.columns if c not in ["Season", "TeamID"]]

    def _fill_group_median(series: pd.Series) -> pd.Series:
        non_na = series.dropna()
        if non_na.empty:
            return series.fillna(0.0)
        return series.fillna(float(non_na.median()))

    if numeric_cols:
        out[numeric_cols] = out.groupby("Season")[numeric_cols].transform(_fill_group_median)
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def _extend_features_to_target_season(
    team_features: pd.DataFrame,
    team_ids: np.ndarray,
    season: int,
) -> pd.DataFrame:
    """Create pseudo-season rows from each team's latest historical season."""
    out = team_features.copy()
    if out.empty:
        return out

    required_cols = ["Season", "TeamID"]
    if not set(required_cols).issubset(out.columns):
        return out

    numeric_cols = [c for c in out.columns if c not in required_cols]
    latest = (
        out.sort_values(["TeamID", "Season"]).groupby("TeamID", as_index=False).tail(1).copy()
    )
    latest["Season"] = int(season)
    latest = latest[latest["TeamID"].isin(team_ids)].copy()

    missing = np.setdiff1d(team_ids, latest["TeamID"].astype(int).to_numpy())
    if len(missing) > 0:
        fill_vals = {c: float(out[c].median()) if c in out.columns else 0.0 for c in numeric_cols}
        miss_df = pd.DataFrame({"Season": int(season), "TeamID": missing})
        for c in numeric_cols:
            miss_df[c] = fill_vals.get(c, 0.0)
        latest = pd.concat([latest, miss_df], ignore_index=True)

    out = pd.concat([out, latest], ignore_index=True)
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def _load_pair_features(features_dir: Path, stem: str, season: int) -> pd.DataFrame:
    path = features_dir / stem
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keys = ["Season", "Team1", "Team2"]
    if not set(keys).issubset(df.columns):
        return pd.DataFrame()
    df = df[df["Season"].astype(int) <= int(season)].copy()
    if df.empty:
        return pd.DataFrame()
    feature_cols = [c for c in df.columns if c not in keys]
    if not feature_cols:
        return pd.DataFrame()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if df.duplicated(subset=keys).any() and numeric_cols:
        df = df.groupby(keys, as_index=False)[numeric_cols].mean()
        feature_cols = [c for c in df.columns if c not in keys]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df


def _build_train_and_pred_frames(
    *,
    pairs_gender: pd.DataFrame,
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    team_features: pd.DataFrame,
    pair_features: list[pd.DataFrame],
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    games = build_game_training_rows(reg_compact, tourney_compact)
    games = games[games["Season"].astype(int) < int(season)].copy()

    train_df = build_matchup_matrix(games, team_features)

    pred_games = pairs_gender[["Season", "Team1", "Team2"]].copy()
    pred_games["y_true"] = 0
    pred_games["Margin"] = 0.0
    pred_games["IsTourney"] = 1
    pred_df = build_matchup_matrix(pred_games, team_features)

    added_feature_cols: list[str] = []
    for pf in pair_features:
        if pf.empty:
            continue
        keys = ["Season", "Team1", "Team2"]
        fcols = [c for c in pf.columns if c not in keys]
        if not fcols:
            continue

        before_cols = set(train_df.columns)
        train_df = train_df.merge(pf, on=keys, how="left")
        pred_df = pred_df.merge(pf, on=keys, how="left")
        for c in train_df.columns:
            if c not in before_cols and c not in keys:
                added_feature_cols.append(c)

    for c in set(added_feature_cols):
        if c in train_df.columns:
            train_df[c] = train_df[c].fillna(0.0)
        if c in pred_df.columns:
            pred_df[c] = pred_df[c].fillna(0.0)

    feature_cols = feature_columns_for_training(train_df)
    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = 0.0

    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    pred_df[feature_cols] = pred_df[feature_cols].fillna(0.0)
    return train_df, pred_df, feature_cols


def _parse_calibration_label(label: str) -> CalibrationSpec | None:
    m = re.search(r"_cal_([^_]+)_(all|tournament_only)_s(\d{2})$", str(label).strip())
    if not m:
        return None
    method = m.group(1).strip().lower()
    scope = m.group(2).strip().lower()
    shrink = float(int(m.group(3))) / 100.0
    return CalibrationSpec(method=method, scope=scope, shrink=shrink)


def _load_calibration_spec(eval_dir: Path, gender: str, model_name: str) -> CalibrationSpec | None:
    best_path = eval_dir / f"calibration_best_{gender.lower()}.csv"
    if not best_path.exists():
        return None

    best_df = pd.read_csv(best_path)
    if best_df.empty or "best_model_label" not in best_df.columns:
        return None

    scoped = best_df
    if "source_model" in best_df.columns:
        scoped = best_df[best_df["source_model"].astype(str).str.lower() == model_name.lower()].copy()
    if scoped.empty:
        return None
    if "best_tournament_brier" in scoped.columns:
        scoped = scoped.sort_values("best_tournament_brier", ascending=True)

    label = str(scoped["best_model_label"].iloc[0])
    return _parse_calibration_label(label)


def _fit_calibration_transform(
    oof_df: pd.DataFrame,
    spec: CalibrationSpec | None,
    season: int,
    min_rows: int = 200,
) -> Callable[[np.ndarray], np.ndarray]:
    if spec is None:
        return lambda p: np.asarray(p, dtype=float)

    req = {"Season", "y_true", "pred"}
    if not req.issubset(oof_df.columns):
        return lambda p: np.asarray(p, dtype=float)

    hist = oof_df[oof_df["Season"].astype(int) < int(season)].copy()
    if hist.empty:
        return lambda p: np.asarray(p, dtype=float)

    train_cal = hist
    if spec.scope == "tournament_only" and "IsTourney" in hist.columns:
        train_cal = hist[hist["IsTourney"].astype(int) == 1].copy()

    if len(train_cal) < min_rows or train_cal["y_true"].nunique(dropna=True) < 2:
        train_cal = hist
    if len(train_cal) < min_rows or train_cal["y_true"].nunique(dropna=True) < 2:
        return lambda p: np.asarray(p, dtype=float)

    model = fit_calibrator(
        spec.method,
        pred=np.asarray(train_cal["pred"], dtype=float),
        y_true=np.asarray(train_cal["y_true"], dtype=int),
    )

    def _transform(pred: np.ndarray) -> np.ndarray:
        p = np.asarray(model.predict(np.asarray(pred, dtype=float)), dtype=float)
        if spec.shrink <= 0.0:
            return p
        return np.asarray((1.0 - spec.shrink) * p + spec.shrink * 0.5, dtype=float)

    return _transform


def _predict_with_base_model(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> np.ndarray:
    factories = build_base_model_factories(include_extras=True)
    if model_name not in factories:
        raise ValueError(
            f"Requested model '{model_name}' is unavailable in build_base_model_factories()."
        )

    model = factories[model_name]()
    train_target = "y_true"
    fit_df = train_df
    if model_name in {"xgb_margin_m", "xgb_margin_w"} and "Margin" in train_df.columns:
        train_target = "Margin"
        if "IsTourney" in train_df.columns:
            fit_df = train_df[train_df["IsTourney"].astype(int) == 0].copy()
            if fit_df.empty:
                fit_df = train_df

    model.fit(fit_df[feature_cols], fit_df[train_target])

    if hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict_proba(pred_df[feature_cols])[:, 1], dtype=float)
    else:
        pred = np.asarray(model.predict(pred_df[feature_cols]), dtype=float)
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


def _fit_stack_from_oof(
    oof_dir: Path,
    gender: str,
    season: int,
    available_models: list[str],
) -> tuple[Ridge, list[str]]:
    parts: list[pd.DataFrame] = []
    for model_name in available_models:
        path = oof_dir / f"oof_{model_name}_{gender.lower()}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        req = {"Season", "Team1", "Team2", "y_true", "pred"}
        if not req.issubset(df.columns):
            continue
        df = df[df["Season"].astype(int) < int(season)].copy()
        if df.empty:
            continue
        keep_cols = [c for c in ["Season", "GameRowID", "Team1", "Team2", "y_true", "IsTourney", "pred"] if c in df.columns]
        part = df[keep_cols].copy()
        part["model"] = model_name
        parts.append(part)

    if not parts:
        raise RuntimeError("No base OOF files available to refit stack meta-learner.")

    long_oof = pd.concat(parts, ignore_index=True)
    stack_wide, pred_cols = build_stack_features(long_oof)
    pred_cols = [c for c in pred_cols if c in available_models]
    if not pred_cols:
        raise RuntimeError("Stack meta-learner has no usable base prediction columns.")

    x = stack_wide[pred_cols].fillna(0.5)
    y = stack_wide["y_true"].astype(float)
    weights = np.ones(len(stack_wide), dtype=float)
    if "IsTourney" in stack_wide.columns:
        weights = np.where(stack_wide["IsTourney"].astype(int).to_numpy() == 1, 1.0, 1.0)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y, sample_weight=weights)
    return ridge, pred_cols


def _predict_stack(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    oof_dir: Path,
    gender: str,
    season: int,
) -> np.ndarray:
    factories = build_base_model_factories(include_extras=True)
    base_models = sorted(
        {
            p.stem[len("oof_") : -len(f"_{gender.lower()}")]
            for p in oof_dir.glob(f"oof_*_{gender.lower()}.csv")
            if p.stem.startswith("oof_")
            and "stack" not in p.stem
            and "_cal_" not in p.stem
        }
    )
    usable_models = [m for m in base_models if m in factories]
    if not usable_models:
        raise RuntimeError(f"No usable base models found for stack ({gender}).")

    ridge, stack_cols = _fit_stack_from_oof(
        oof_dir=oof_dir,
        gender=gender,
        season=season,
        available_models=usable_models,
    )

    pred_matrix = pd.DataFrame(index=pred_df.index)
    for model_name in stack_cols:
        pred_matrix[model_name] = _predict_with_base_model(
            train_df=train_df,
            pred_df=pred_df,
            feature_cols=feature_cols,
            model_name=model_name,
        )

    pred = np.asarray(ridge.predict(pred_matrix[stack_cols].fillna(0.5)), dtype=float)
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


def regime_a_probability(
    pairs: pd.DataFrame,
    m_model_oof_path: Path,
    w_model_oof_path: Path,
    team_features_m: pd.DataFrame,
    team_features_w: pd.DataFrame,
    season: int,
    config: SubmissionConfig,
) -> pd.DataFrame:
    """Refit selected final models and predict seeded matchup pairs."""
    out = pairs.copy()
    if out.empty:
        out["Pred"] = np.array([], dtype=float)
        return out

    def _model_name_from_oof_path(path: Path, gender_tag: str) -> str:
        name = path.name
        name = name.replace("oof_", "")
        name = name.replace(f"_{gender_tag}.csv", "")
        name = name.replace("_cal_best", "")
        return name

    m_model_name = _model_name_from_oof_path(m_model_oof_path, "m")
    w_model_name = _model_name_from_oof_path(w_model_oof_path, "w")

    reg_m = pd.read_csv(config.data_dir / "MRegularSeasonCompactResults.csv")
    reg_w = pd.read_csv(config.data_dir / "WRegularSeasonCompactResults.csv")
    trn_m = pd.read_csv(config.data_dir / "MNCAATourneyCompactResults.csv")
    trn_w = pd.read_csv(config.data_dir / "WNCAATourneyCompactResults.csv")

    graph_m = _load_pair_features(config.features_dir, "graph_features_m.csv", season)
    graph_w = _load_pair_features(config.features_dir, "graph_features_w.csv", season)
    temporal_m = _load_pair_features(config.features_dir, "temporal_features_m.csv", season)
    temporal_w = _load_pair_features(config.features_dir, "temporal_features_w.csv", season)

    eval_dir = config.eval_dir

    parts: list[pd.DataFrame] = []
    for gender, model_name, oof_path, reg, trn, team_feat, pf in [
        ("M", m_model_name, m_model_oof_path, reg_m, trn_m, team_features_m, [graph_m, temporal_m]),
        ("W", w_model_name, w_model_oof_path, reg_w, trn_w, team_features_w, [graph_w, temporal_w]),
    ]:
        sub_pairs = out[out["Gender"] == gender].copy()
        if sub_pairs.empty:
            continue

        train_df, pred_df, feature_cols = _build_train_and_pred_frames(
            pairs_gender=sub_pairs,
            reg_compact=reg,
            tourney_compact=trn,
            team_features=team_feat,
            pair_features=pf,
            season=season,
        )
        if train_df.empty or pred_df.empty:
            sub_pairs["Pred"] = 0.5
            parts.append(sub_pairs)
            continue

        if model_name == "stack":
            pred = _predict_stack(
                train_df=train_df,
                pred_df=pred_df,
                feature_cols=feature_cols,
                oof_dir=oof_path.parents[0],
                gender=gender,
                season=season,
            )
        else:
            pred = _predict_with_base_model(
                train_df=train_df,
                pred_df=pred_df,
                feature_cols=feature_cols,
                model_name=model_name,
            )

        is_already_calibrated = "cal_best" in oof_path.name
        if oof_path.exists() and not is_already_calibrated:
            oof_df = pd.read_csv(oof_path)
            spec = _load_calibration_spec(eval_dir=eval_dir, gender=gender, model_name=model_name)
            transform = _fit_calibration_transform(oof_df=oof_df, spec=spec, season=season)
            pred = transform(pred)

        sub_pairs["Pred"] = np.asarray(pred, dtype=float)
        parts.append(sub_pairs)

    if not parts:
        out["Pred"] = 0.5
        return out
    return pd.concat(parts, ignore_index=True)


def _pick_oof_path(oof_dir: Path, gender: str, model_name: str) -> Path:
    source = oof_dir / f"oof_{model_name}_{gender}.csv"
    cal_best = oof_dir / f"oof_{model_name}_cal_best_{gender}.csv"
    if source.exists():
        return source
    if cal_best.exists():
        return cal_best
    return source


def _expected_rows_from_sample(data_dir: Path, season: int) -> int | None:
    sample_files = [
        data_dir / "SampleSubmissionStage2.csv",
        data_dir / "SampleSubmission.csv",
    ]
    for p in sample_files:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "ID" not in df.columns:
            continue
        season_ids = df["ID"].astype(str).str.startswith(f"{season}_")
        count = int(season_ids.sum())
        if count > 0:
            return count
    return None


def _safe_submission_path(submissions_dir: Path, season: int, suffix: str) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    clean_suffix = suffix.strip().strip("_")
    stem = f"sub_{day}_{season}"
    if clean_suffix:
        stem += f"_{clean_suffix}"

    candidate = submissions_dir / f"{stem}.csv"
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        alt = submissions_dir / f"{stem}_v{idx}.csv"
        if not alt.exists():
            return alt
        idx += 1


def build_submission(
    data_dir: Path,
    features_dir: Path,
    oof_dir: Path,
    submissions_dir: Path,
    season: int,
    m_model: str = "stack",
    w_model: str = "logreg",
    regime_b_scale: float = 14.0,
    output_suffix: str = "",
    clip_range: tuple = (0.05, 0.95),
) -> Path:
    """Build full phase-8 submission CSV for the selected season."""
    data_dir = Path(data_dir)
    features_dir = Path(features_dir)
    oof_dir = Path(oof_dir)
    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    config = SubmissionConfig(
        data_dir=data_dir,
        features_dir=features_dir,
        oof_dir=oof_dir,
        submissions_dir=submissions_dir,
        eval_dir=ROOT / "eval",
    )

    pairs = build_all_pairs(data_dir=data_dir, season=int(season))
    if pairs.empty:
        raise RuntimeError(f"No matchup pairs found for season={season}.")

    seeded = load_seeds(data_dir=data_dir, season=int(season))
    if not seeded:
        print(f"[PHASE8] WARNING: no seeds found for season={season}; using Regime B for all rows.")

    pairs["IsRegimeA"] = pairs["Team1"].isin(seeded) & pairs["Team2"].isin(seeded)
    regime_a = pairs[pairs["IsRegimeA"]].copy()
    regime_b = pairs[~pairs["IsRegimeA"]].copy()

    m_reg = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
    w_reg = pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv")

    # Leakage guard: build team features using seasons strictly before target season.
    m_basic = build_basic_team_features(m_reg[m_reg["Season"].astype(int) < int(season)].copy())
    w_basic = build_basic_team_features(w_reg[w_reg["Season"].astype(int) < int(season)].copy())

    m_ratings_path = features_dir / "team_ratings_m.parquet"
    w_ratings_path = features_dir / "team_ratings_w.parquet"
    if not m_ratings_path.exists() or not w_ratings_path.exists():
        raise FileNotFoundError(
            "Missing team ratings parquet files in features/. Run baseline ratings refresh first."
        )

    m_ratings = pd.read_parquet(m_ratings_path)
    w_ratings = pd.read_parquet(w_ratings_path)
    m_ratings = m_ratings[m_ratings["Season"].astype(int) < int(season)].copy()
    w_ratings = w_ratings[w_ratings["Season"].astype(int) < int(season)].copy()

    m_team_features = _combine_team_features(m_basic, m_ratings)
    w_team_features = _combine_team_features(w_basic, w_ratings)

    m_target_ids = np.sort(
        pd.unique(
            pd.concat(
                [
                    pairs.loc[pairs["Gender"] == "M", "Team1"],
                    pairs.loc[pairs["Gender"] == "M", "Team2"],
                ],
                ignore_index=True,
            )
        ).astype(int)
    )
    w_target_ids = np.sort(
        pd.unique(
            pd.concat(
                [
                    pairs.loc[pairs["Gender"] == "W", "Team1"],
                    pairs.loc[pairs["Gender"] == "W", "Team2"],
                ],
                ignore_index=True,
            )
        ).astype(int)
    )

    m_team_features = _extend_features_to_target_season(m_team_features, m_target_ids, int(season))
    w_team_features = _extend_features_to_target_season(w_team_features, w_target_ids, int(season))

    ratings_all = pd.concat(
        [
            m_team_features[["Season", "TeamID", "AdjNet"]] if "AdjNet" in m_team_features.columns else pd.DataFrame(columns=["Season", "TeamID", "AdjNet"]),
            w_team_features[["Season", "TeamID", "AdjNet"]] if "AdjNet" in w_team_features.columns else pd.DataFrame(columns=["Season", "TeamID", "AdjNet"]),
        ],
        ignore_index=True,
    )

    regime_b_out = regime_b_probability(
        pairs=regime_b,
        team_ratings=ratings_all,
        scale=float(regime_b_scale),
    )

    m_oof_path = _pick_oof_path(oof_dir=oof_dir, gender="m", model_name=m_model)
    w_oof_path = _pick_oof_path(oof_dir=oof_dir, gender="w", model_name=w_model)
    if not m_oof_path.exists():
        raise FileNotFoundError(f"Men OOF source not found: {m_oof_path}")
    if not w_oof_path.exists():
        raise FileNotFoundError(f"Women OOF source not found: {w_oof_path}")

    regime_a_out = regime_a_probability(
        pairs=regime_a,
        m_model_oof_path=m_oof_path,
        w_model_oof_path=w_oof_path,
        team_features_m=m_team_features,
        team_features_w=w_team_features,
        season=int(season),
        config=config,
    )

    final = pd.concat([regime_a_out, regime_b_out], ignore_index=True)
    final["Pred"] = _clip_prob(np.asarray(final["Pred"], dtype=float), clip_range=clip_range)
    final = final.sort_values("ID").reset_index(drop=True)
    out = final[["ID", "Pred"]].copy()

    expected = _expected_rows_from_sample(data_dir=data_dir, season=int(season))
    warning_suffix = ""
    if expected is not None:
        row_diff = abs(len(out) - int(expected))
        if row_diff > 100:
            warning_suffix = "_ROWCOUNT_WARNING"
            print(
                f"[PHASE8] WARNING: row count mismatch vs sample for season={season}: "
                f"got={len(out):,}, expected={expected:,}, diff={row_diff:,}",
            )

    full_suffix = output_suffix.strip().strip("_")
    if warning_suffix:
        full_suffix = (full_suffix + warning_suffix).strip("_")

    save_path = _safe_submission_path(submissions_dir=submissions_dir, season=int(season), suffix=full_suffix)
    out.to_csv(save_path, index=False, float_format="%.6f")

    total = len(final)
    regime_a_n = len(regime_a_out)
    regime_b_n = len(regime_b_out)
    men_a = len(regime_a_out[regime_a_out["Gender"] == "M"])
    women_a = len(regime_a_out[regime_a_out["Gender"] == "W"])

    print(f"Submission saved: {save_path}")
    print(f"Total rows: {total}")
    print(f"Regime A (seeded pairs): {regime_a_n}  ({(100.0 * regime_a_n / max(total, 1)):.1f}%)")
    print(f"Regime B (all others): {regime_b_n}  ({(100.0 * regime_b_n / max(total, 1)):.1f}%)")
    print(
        "Pred distribution: "
        f"mean={out['Pred'].mean():.3f} "
        f"std={out['Pred'].std(ddof=0):.3f} "
        f"min={out['Pred'].min():.3f} "
        f"max={out['Pred'].max():.3f}"
    )
    print(f"Men Regime A rows: {men_a}")
    print(f"Women Regime A rows: {women_a}")

    return save_path
