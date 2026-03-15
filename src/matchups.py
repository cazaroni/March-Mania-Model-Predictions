"""Matchup row builders with raw, diff, interaction, and rating features."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def canonical_matchup(df: pd.DataFrame, winner_col: str, loser_col: str) -> pd.DataFrame:
    """Build canonical Team1/Team2 and target where Team1 is lower TeamID."""
    out = df.copy()
    out["Team1"] = out[[winner_col, loser_col]].min(axis=1)
    out["Team2"] = out[[winner_col, loser_col]].max(axis=1)
    out["y_true"] = (out[winner_col] == out["Team1"]).astype(int)
    return out


def build_game_training_rows(
    reg_compact: pd.DataFrame,
    tourney_compact: pd.DataFrame,
) -> pd.DataFrame:
    """Combine regular + tournament results into one training frame."""
    reg = canonical_matchup(reg_compact[["Season", "DayNum", "WTeamID", "LTeamID"]], "WTeamID", "LTeamID")
    reg["IsTourney"] = 0

    trn = canonical_matchup(tourney_compact[["Season", "DayNum", "WTeamID", "LTeamID"]], "WTeamID", "LTeamID")
    trn["IsTourney"] = 1

    return pd.concat(
        [reg[["Season", "Team1", "Team2", "y_true", "IsTourney"]], trn[["Season", "Team1", "Team2", "y_true", "IsTourney"]]],
        ignore_index=True,
    )


def _all_numeric_feature_cols(team_features: pd.DataFrame) -> list[str]:
    return [
        c
        for c in team_features.columns
        if c not in ["Season", "TeamID"] and pd.api.types.is_numeric_dtype(team_features[c])
    ]


def build_matchup_matrix(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build T1_*, T2_*, Diff_*, Interact_*, and Rating_* features."""
    f = team_features.copy()
    num_cols = _all_numeric_feature_cols(f)

    t1 = f[["Season", "TeamID"] + num_cols].copy()
    t2 = t1.copy()

    x = games.merge(t1, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    x = x.merge(
        t2,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("_T1", "_T2"),
    ).drop(columns=["TeamID"])

    for c in num_cols:
        c1, c2 = f"{c}_T1", f"{c}_T2"
        if c1 in x.columns and c2 in x.columns:
            x[f"Diff_{c}"] = x[c1] - x[c2]

    x = add_interaction_features(x)
    x = add_structural_uncertainty_features(x)
    return x


def _col_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(np.zeros(len(df), dtype=float), index=df.index)


def add_interaction_features(x: pd.DataFrame) -> pd.DataFrame:
    """Add core interaction features between offense and opposing defense."""
    x = x.copy()
    x["Interact_offeFG_vs_defeFG_T1"] = _col_or_zero(x, "off_eFG_T1") - _col_or_zero(x, "def_eFG_T2")
    x["Interact_offeFG_vs_defeFG_T2"] = _col_or_zero(x, "off_eFG_T2") - _col_or_zero(x, "def_eFG_T1")

    x["Interact_offTOV_vs_defTOV_T1"] = _col_or_zero(x, "off_TOV_T1") - _col_or_zero(x, "def_TOV_T2")
    x["Interact_offTOV_vs_defTOV_T2"] = _col_or_zero(x, "off_TOV_T2") - _col_or_zero(x, "def_TOV_T1")

    x["Interact_offORB_vs_defDRB_T1"] = _col_or_zero(x, "off_ORB_T1") - _col_or_zero(x, "def_DRB_T2")
    x["Interact_offORB_vs_defDRB_T2"] = _col_or_zero(x, "off_ORB_T2") - _col_or_zero(x, "def_DRB_T1")

    x["Interact_FTR_T1_vs_allowed_T2"] = _col_or_zero(x, "FTR_T1") - _col_or_zero(x, "FTR_allowed_T2")
    x["Interact_3PArate_T1_vs_allowed_T2"] = _col_or_zero(x, "ThreePAR_T1") - _col_or_zero(x, "ThreePAR_allowed_T2")
    return x


def add_structural_uncertainty_features(x: pd.DataFrame) -> pd.DataFrame:
    """Add structural and uncertainty gap/clash features."""
    x = x.copy()
    x["Interact_PaceClash"] = (_col_or_zero(x, "AdjPace_T1") - _col_or_zero(x, "AdjPace_T2")).abs()
    x["Interact_VolatilityClash"] = (_col_or_zero(x, "Volatility_T1") - _col_or_zero(x, "Volatility_T2")).abs()
    x["Interact_RecentFormClash"] = _col_or_zero(x, "Last10WinPct_T1") - _col_or_zero(x, "Last10WinPct_T2")
    x["Interact_ConferenceMismatch"] = (_col_or_zero(x, "ConfStrength_T1") - _col_or_zero(x, "ConfStrength_T2")).abs()

    x["Rating_Gap"] = (_col_or_zero(x, "Elo_T1") - _col_or_zero(x, "Elo_T2")).abs()
    x["Rating_RankDispersionGap"] = (_col_or_zero(x, "RankStd_T1") - _col_or_zero(x, "RankStd_T2")).abs()
    x["Rating_LuckResidualGap"] = (_col_or_zero(x, "LuckResidual_T1") - _col_or_zero(x, "LuckResidual_T2")).abs()
    x["Rating_ConsistencyGap"] = (_col_or_zero(x, "Consistency_T1") - _col_or_zero(x, "Consistency_T2")).abs()
    return x


def feature_columns_for_training(df: pd.DataFrame) -> list[str]:
    """Return usable training features with required families (including graph features)."""
    keep_prefixes = (
        "T1_",
        "T2_",
        "Diff_",
        "Interact_",
        "Rating_",
        "Embed",
        "Cluster",
        "Neighbor",
        "Archetype",
        "LSTM_",
        "GRU_",
    )
    cols = [c for c in df.columns if any(c.startswith(p) for p in keep_prefixes) and c not in ["Season", "Team1", "Team2"]]
    return cols
