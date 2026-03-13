"""Baseline season feature engineering for NCAA men and women."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load required Kaggle NCAA files for baseline + advanced phases."""
    d: Dict[str, pd.DataFrame] = {}

    d["m_reg"] = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
    d["w_reg"] = pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv")
    d["m_reg_det"] = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    d["w_reg_det"] = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")

    d["m_tourney"] = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    d["w_tourney"] = pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv")

    d["m_team_conf"] = pd.read_csv(data_dir / "MTeamConferences.csv")
    d["w_team_conf"] = pd.read_csv(data_dir / "WTeamConferences.csv")

    d["m_massey"] = pd.read_csv(data_dir / "MMasseyOrdinals.csv")
    d["sub_stage2"] = pd.read_csv(data_dir / "SampleSubmissionStage2.csv")
    return d


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / np.where(den == 0, np.nan, den)


def _team_game_table(reg_compact: pd.DataFrame) -> pd.DataFrame:
    win_side = reg_compact[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppID", "WScore": "Pts", "LScore": "OppPts"}
    )
    win_side["Win"] = 1

    loss_side = reg_compact[["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppID", "LScore": "Pts", "WScore": "OppPts"}
    )
    loss_side["Win"] = 0

    g = pd.concat([win_side, loss_side], ignore_index=True)
    g["Margin"] = g["Pts"] - g["OppPts"]
    return g


def build_basic_team_features(reg_compact: pd.DataFrame) -> pd.DataFrame:
    """Build simple season team aggregates for baseline modeling."""
    g = _team_game_table(reg_compact)

    by_team = (
        g.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            Games=("Win", "size"),
            Wins=("Win", "sum"),
            PointsFor=("Pts", "sum"),
            PointsAllowed=("OppPts", "sum"),
            AvgMargin=("Margin", "mean"),
            MarginStd=("Margin", "std"),
            RecentWinPct=("Win", "mean"),
        )
        .fillna({"MarginStd": 0.0})
    )

    by_team["Losses"] = by_team["Games"] - by_team["Wins"]
    by_team["WinPct"] = _safe_div(by_team["Wins"], by_team["Games"]).fillna(0.0)
    by_team["PPG"] = _safe_div(by_team["PointsFor"], by_team["Games"]).fillna(0.0)
    by_team["PAPG"] = _safe_div(by_team["PointsAllowed"], by_team["Games"]).fillna(0.0)

    g = g.sort_values(["Season", "TeamID", "DayNum"])
    last10 = g.groupby(["Season", "TeamID"]).tail(10)
    recent = (
        last10.groupby(["Season", "TeamID"], as_index=False)
        .agg(Last10WinPct=("Win", "mean"), Last10Margin=("Margin", "mean"), Volatility=("Margin", "std"))
        .fillna({"Volatility": 0.0})
    )

    out = by_team.merge(recent, on=["Season", "TeamID"], how="left")
    out[["Last10WinPct", "Last10Margin", "Volatility"]] = out[["Last10WinPct", "Last10Margin", "Volatility"]].fillna(0.0)
    out["Consistency"] = 1.0 / (1.0 + out["Volatility"].abs())
    return out


def parse_submission_ids(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Parse Kaggle submission IDs into season/team columns."""
    out = sub_df.copy()
    parts = out["ID"].str.split("_", expand=True)
    out["Season"] = parts[0].astype(int)
    out["Team1"] = parts[1].astype(int)
    out["Team2"] = parts[2].astype(int)
    out["Gender"] = np.where(out["Team1"] < 2000, "M", "W")
    return out
