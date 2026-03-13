"""Universal team ratings by season for men and women."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / np.where(den == 0, np.nan, den)


def _all_teams_by_season(reg_compact: pd.DataFrame) -> pd.DataFrame:
    w = reg_compact[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    l = reg_compact[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
    out = pd.concat([w, l], ignore_index=True).drop_duplicates().sort_values(["Season", "TeamID"])
    return out.reset_index(drop=True)


def compute_elo_features(reg_compact: pd.DataFrame, k_base: float = 20.0, home_adv: float = 50.0) -> pd.DataFrame:
    """Compute Elo-style seasonal features using margin multiplier log(abs(diff)+1)."""
    games = reg_compact[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]].copy()
    games = games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    teams = _all_teams_by_season(reg_compact)
    records: list[dict[str, float | int]] = []

    for season, sg in games.groupby("Season"):
        rating: Dict[int, float] = {int(t): 1500.0 for t in teams.loc[teams["Season"] == season, "TeamID"]}
        team_rows: list[dict[str, float | int]] = []

        max_day = int(sg["DayNum"].max())
        for row in sg.itertuples(index=False):
            w = int(row.WTeamID)
            l = int(row.LTeamID)
            day = int(row.DayNum)
            rw = rating.get(w, 1500.0)
            rl = rating.get(l, 1500.0)

            loc = str(row.WLoc)
            if loc == "H":
                rw_home, rl_home = rw + home_adv, rl
            elif loc == "A":
                rw_home, rl_home = rw, rl + home_adv
            else:
                rw_home, rl_home = rw, rl

            exp_w = 1.0 / (1.0 + 10 ** ((rl_home - rw_home) / 400.0))
            score_diff = int(row.WScore) - int(row.LScore)
            margin_mult = np.log(abs(score_diff) + 1.0)
            delta = k_base * margin_mult * (1.0 - exp_w)

            rating[w] = rw + delta
            rating[l] = rl - delta

            team_rows.append({"Season": season, "TeamID": w, "DayNum": day, "PostElo": rating[w], "Loc": loc})
            team_rows.append({"Season": season, "TeamID": l, "DayNum": day, "PostElo": rating[l], "Loc": "H" if loc == "A" else ("A" if loc == "H" else "N")})

        sg_team = pd.DataFrame(team_rows)
        if sg_team.empty:
            continue

        last30_cut = max_day - 30
        recent = sg_team[sg_team["DayNum"] >= last30_cut]

        season_elo = sg_team.groupby(["Season", "TeamID"], as_index=False).agg(
            Elo=("PostElo", "last"),
            NeutralElo=("PostElo", "mean"),
        )

        elo_last30 = recent.groupby(["Season", "TeamID"], as_index=False).agg(EloLast30=("PostElo", "mean"))
        season_elo = season_elo.merge(elo_last30, on=["Season", "TeamID"], how="left")
        season_elo["EloLast30"] = season_elo["EloLast30"].fillna(season_elo["Elo"])
        season_elo["EloDelta"] = season_elo["Elo"] - 1500.0

        home_away = (
            sg_team.assign(is_home=sg_team["Loc"].eq("H").astype(int), is_away=sg_team["Loc"].eq("A").astype(int))
            .groupby(["Season", "TeamID"], as_index=False)
            .agg(home_games=("is_home", "sum"), away_games=("is_away", "sum"), HomeAdjElo=("PostElo", "mean"))
        )
        season_elo = season_elo.merge(home_away[["Season", "TeamID", "HomeAdjElo"]], on=["Season", "TeamID"], how="left")
        records.append(season_elo)

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["Season", "TeamID"])
    return out


def _fit_bt_single_season(games: pd.DataFrame, team_ids: np.ndarray) -> pd.Series:
    """Fit Bradley-Terry rating via pairwise logistic regression for one season."""
    if games.empty:
        return pd.Series(0.0, index=team_ids)

    idx_map = {int(t): i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)

    rows = []
    cols = []
    vals = []
    y = []

    r = 0
    for row in games.itertuples(index=False):
        w = idx_map[int(row.WTeamID)]
        l = idx_map[int(row.LTeamID)]

        rows.extend([r, r])
        cols.extend([w, l])
        vals.extend([1.0, -1.0])
        y.append(1)
        r += 1

        rows.extend([r, r])
        cols.extend([l, w])
        vals.extend([1.0, -1.0])
        y.append(0)
        r += 1

    x = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_teams))
    model = LogisticRegression(
        fit_intercept=False,
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(x, np.asarray(y, dtype=int))
    return pd.Series(model.coef_.ravel(), index=team_ids)


def compute_bt_features(reg_compact: pd.DataFrame, team_conf: pd.DataFrame) -> pd.DataFrame:
    """Compute BT, BT_Recent, BT_ConfAdj per season/team."""
    reg = reg_compact[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    out_parts = []

    for season, sg in reg.groupby("Season"):
        team_ids = np.sort(pd.unique(pd.concat([sg["WTeamID"], sg["LTeamID"]], ignore_index=True))).astype(int)
        bt = _fit_bt_single_season(sg, team_ids)

        cutoff = int(sg["DayNum"].max()) - 30
        bt_recent = _fit_bt_single_season(sg[sg["DayNum"] >= cutoff], team_ids)

        df = pd.DataFrame({"Season": season, "TeamID": team_ids, "BT": bt.values, "BT_Recent": bt_recent.values})
        out_parts.append(df)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["Season", "TeamID", "BT", "BT_Recent"])
    conf = team_conf[["Season", "TeamID", "ConfAbbrev"]].copy()
    out = out.merge(conf, on=["Season", "TeamID"], how="left")
    conf_mean = out.groupby(["Season", "ConfAbbrev"], as_index=False)["BT"].mean().rename(columns={"BT": "ConfBTMean"})
    out = out.merge(conf_mean, on=["Season", "ConfAbbrev"], how="left")
    out["BT_ConfAdj"] = out["BT"] - out["ConfBTMean"].fillna(0.0)
    return out[["Season", "TeamID", "BT", "BT_Recent", "BT_ConfAdj"]]


def compute_efficiency_features(reg_det: pd.DataFrame) -> pd.DataFrame:
    """Compute adjusted efficiency and four-factor style features."""
    w = reg_det[[
        "Season", "DayNum", "WTeamID", "WScore", "LScore", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WTO", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LTO",
    ]].copy()
    w.columns = [
        "Season", "DayNum", "TeamID", "Pts", "OppPts", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "TO", "OppFGM", "OppFGA", "OppFGM3", "OppFGA3", "OppFTM", "OppFTA", "OppOR", "OppDR", "OppTO",
    ]

    l = reg_det[[
        "Season", "DayNum", "LTeamID", "LScore", "WScore", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LTO", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WTO",
    ]].copy()
    l.columns = w.columns

    g = pd.concat([w, l], ignore_index=True)
    poss = g["FGA"] - g["OR"] + g["TO"] + 0.475 * g["FTA"]
    opp_poss = g["OppFGA"] - g["OppOR"] + g["OppTO"] + 0.475 * g["OppFTA"]

    g["poss"] = poss
    g["opp_poss"] = opp_poss
    g["OffRating"] = 100.0 * _safe_div(g["Pts"], g["poss"])
    g["DefRating"] = 100.0 * _safe_div(g["OppPts"], g["opp_poss"])
    g["AdjNet"] = g["OffRating"] - g["DefRating"]
    g["AdjPace"] = (g["poss"] + g["opp_poss"]) / 2.0

    g["off_eFG"] = _safe_div(g["FGM"] + 0.5 * g["FGM3"], g["FGA"])
    g["def_eFG"] = _safe_div(g["OppFGM"] + 0.5 * g["OppFGM3"], g["OppFGA"])
    g["off_TOV"] = _safe_div(g["TO"], g["FGA"] + 0.44 * g["FTA"] + g["TO"])
    g["def_TOV"] = _safe_div(g["OppTO"], g["OppFGA"] + 0.44 * g["OppFTA"] + g["OppTO"])
    g["off_ORB"] = _safe_div(g["OR"], g["OR"] + g["OppDR"])
    g["def_DRB"] = _safe_div(g["DR"], g["DR"] + g["OppOR"])
    g["FTR"] = _safe_div(g["FTM"], g["FGA"])
    g["FTR_allowed"] = _safe_div(g["OppFTM"], g["OppFGA"])
    g["ThreePAR"] = _safe_div(g["FGA3"], g["FGA"])
    g["ThreePAR_allowed"] = _safe_div(g["OppFGA3"], g["OppFGA"])

    g["Margin"] = g["Pts"] - g["OppPts"]

    exp = 11.5
    g["PythagWinComp"] = (g["Pts"] ** exp) / ((g["Pts"] ** exp) + (g["OppPts"] ** exp))
    g["ActualWin"] = (g["Pts"] > g["OppPts"]).astype(int)

    agg = (
        g.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            OffRating=("OffRating", "mean"),
            DefRating=("DefRating", "mean"),
            AdjNet=("AdjNet", "mean"),
            AdjPace=("AdjPace", "mean"),
            off_eFG=("off_eFG", "mean"),
            def_eFG=("def_eFG", "mean"),
            off_TOV=("off_TOV", "mean"),
            def_TOV=("def_TOV", "mean"),
            off_ORB=("off_ORB", "mean"),
            def_DRB=("def_DRB", "mean"),
            FTR=("FTR", "mean"),
            FTR_allowed=("FTR_allowed", "mean"),
            ThreePAR=("ThreePAR", "mean"),
            ThreePAR_allowed=("ThreePAR_allowed", "mean"),
            PythagWinPct=("PythagWinComp", "mean"),
            ActualWinPct=("ActualWin", "mean"),
            Volatility=("Margin", "std"),
        )
        .fillna({"Volatility": 0.0})
    )

    agg["LuckResidual"] = agg["ActualWinPct"] - agg["PythagWinPct"]
    agg["Consistency"] = 1.0 / (1.0 + agg["Volatility"].abs())
    return agg


def compute_conference_strength(team_conf: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    """Conference strength feature from team strength proxy."""
    x = team_conf[["Season", "TeamID", "ConfAbbrev"]].merge(
        team_strength[["Season", "TeamID", "AdjNet"]], on=["Season", "TeamID"], how="left"
    )
    conf = x.groupby(["Season", "ConfAbbrev"], as_index=False).agg(ConfStrength=("AdjNet", "mean"))
    out = x.merge(conf, on=["Season", "ConfAbbrev"], how="left")
    return out[["Season", "TeamID", "ConfStrength"]]


def compute_massey_aggregation(massey: pd.DataFrame) -> pd.DataFrame:
    """Aggregate men Massey ordinals at RankingDayNum=133."""
    m = massey[massey["RankingDayNum"] == 133].copy()
    if m.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "RankMean", "RankMedian", "RankStd", "RankMin", "RankMax"])

    agg = (
        m.groupby(["Season", "TeamID"], as_index=False)["OrdinalRank"]
        .agg(RankMean="mean", RankMedian="median", RankStd="std", RankMin="min", RankMax="max")
        .fillna({"RankStd": 0.0})
    )
    return agg


def build_universal_team_ratings(
    *,
    reg_compact: pd.DataFrame,
    reg_detailed: pd.DataFrame,
    team_conf: pd.DataFrame,
    massey: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create a season/team rating table covering all teams in regular season data."""
    teams = _all_teams_by_season(reg_compact)

    elo = compute_elo_features(reg_compact)
    bt = compute_bt_features(reg_compact, team_conf)
    eff = compute_efficiency_features(reg_detailed)
    conf = compute_conference_strength(team_conf, eff)

    out = teams.merge(elo, on=["Season", "TeamID"], how="left")
    out = out.merge(bt, on=["Season", "TeamID"], how="left")
    out = out.merge(eff, on=["Season", "TeamID"], how="left")
    out = out.merge(conf, on=["Season", "TeamID"], how="left")

    if massey is not None:
        out = out.merge(compute_massey_aggregation(massey), on=["Season", "TeamID"], how="left")

    numeric_cols = [c for c in out.columns if c not in ["Season", "TeamID"]]
    out[numeric_cols] = out[numeric_cols].astype(float)
    out[numeric_cols] = out.groupby("Season")[numeric_cols].transform(lambda s: s.fillna(s.median()))
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def build_and_save_ratings(
    data_dir: Path,
    features_dir: Path,
    *,
    reg_compact_file: str,
    reg_detailed_file: str,
    team_conf_file: str,
    output_file: str,
    massey_file: str | None = None,
) -> pd.DataFrame:
    """Load source files, build universal ratings, save parquet."""
    reg_compact = pd.read_csv(data_dir / reg_compact_file)
    reg_detailed = pd.read_csv(data_dir / reg_detailed_file)
    team_conf = pd.read_csv(data_dir / team_conf_file)
    massey = pd.read_csv(data_dir / massey_file) if massey_file else None

    ratings = build_universal_team_ratings(
        reg_compact=reg_compact,
        reg_detailed=reg_detailed,
        team_conf=team_conf,
        massey=massey,
    )
    features_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(features_dir / output_file, index=False)
    return ratings
