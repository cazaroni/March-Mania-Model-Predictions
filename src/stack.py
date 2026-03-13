"""Leakage-safe stacking on rolling OOF base predictions."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def build_stack_features(base_oof: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot model-level OOF rows to wide matrix for meta-model training."""
    key_cols = ["Season", "Team1", "Team2", "y_true"]
    if "IsTourney" in base_oof.columns:
        key_cols.append("IsTourney")

    wide = (
        base_oof.pivot_table(index=key_cols, columns="model", values="pred", aggfunc="mean")
        .reset_index()
        .rename_axis(columns=None)
    )
    pred_cols = [c for c in wide.columns if c not in key_cols]
    return wide, pred_cols


def rolling_stack_oof(
    stack_df: pd.DataFrame,
    *,
    season_col: str = "Season",
    y_col: str = "y_true",
    feature_cols: list[str],
) -> pd.DataFrame:
    """Train ridge stacker in rolling, leakage-safe mode."""
    seasons = sorted(stack_df[season_col].astype(int).unique())
    out_parts: List[pd.DataFrame] = []

    for valid_season in seasons[1:]:
        tr = stack_df[stack_df[season_col].astype(int) < valid_season].copy()
        va = stack_df[stack_df[season_col].astype(int) == valid_season].copy()
        if tr.empty or va.empty:
            continue

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(tr[feature_cols].fillna(0.5), tr[y_col].astype(float))
        pred = np.clip(ridge.predict(va[feature_cols].fillna(0.5)), 1e-6, 1 - 1e-6)

        oof = va[["Season", "Team1", "Team2", y_col] + (["IsTourney"] if "IsTourney" in va.columns else [])].copy()
        oof["pred"] = pred
        oof["model"] = "stack"
        out_parts.append(oof)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
