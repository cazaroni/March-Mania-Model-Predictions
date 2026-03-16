"""Phase 7: Temporal sequence embeddings (GRU) for NCAA team-season trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


INPUT_DIM = 15
MIN_GAMES_FOR_EMBED = 5


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.divide(num, np.where(den == 0.0, np.nan, den))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _loc_onehot_team_perspective(wloc: pd.Series, is_winner_side: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loc = wloc.fillna("N").astype(str).str.upper().to_numpy()
    loc = np.where(np.isin(loc, ["H", "A", "N"]), loc, "N")

    if is_winner_side:
        team_loc = loc
    else:
        team_loc = np.where(loc == "H", "A", np.where(loc == "A", "H", "N"))

    home = (team_loc == "H").astype(np.float32)
    away = (team_loc == "A").astype(np.float32)
    neutral = (team_loc == "N").astype(np.float32)
    return home, away, neutral


def _team_game_rows_from_detailed(reg_detailed: pd.DataFrame) -> pd.DataFrame:
    """Expand detailed regular-season games to one row per team-game in team perspective."""

    win = reg_detailed.copy()
    win_home, win_away, win_neutral = _loc_onehot_team_perspective(win["WLoc"], is_winner_side=True)

    wfgm = win["WFGM"].to_numpy(dtype=np.float32)
    wfga = win["WFGA"].to_numpy(dtype=np.float32)
    wfgm3 = win["WFGM3"].to_numpy(dtype=np.float32)
    wfta = win["WFTA"].to_numpy(dtype=np.float32)
    wto = win["WTO"].to_numpy(dtype=np.float32)
    wor = win["WOR"].to_numpy(dtype=np.float32)
    wdr = win["WDR"].to_numpy(dtype=np.float32)
    wscore = win["WScore"].to_numpy(dtype=np.float32)

    lfgm = win["LFGM"].to_numpy(dtype=np.float32)
    lfga = win["LFGA"].to_numpy(dtype=np.float32)
    lfgm3 = win["LFGM3"].to_numpy(dtype=np.float32)
    lfta = win["LFTA"].to_numpy(dtype=np.float32)
    lto = win["LTO"].to_numpy(dtype=np.float32)
    lor = win["LOR"].to_numpy(dtype=np.float32)
    ldr = win["LDR"].to_numpy(dtype=np.float32)
    lscore = win["LScore"].to_numpy(dtype=np.float32)

    w_efg = _safe_div(wfgm + 0.5 * wfgm3, wfga)
    l_efg = _safe_div(lfgm + 0.5 * lfgm3, lfga)
    w_tov_rate = _safe_div(wto, wfga + 0.44 * wfta + wto)
    l_tov_rate = _safe_div(lto, lfga + 0.44 * lfta + lto)
    w_orb = _safe_div(wor, wor + ldr)
    w_drb = _safe_div(wdr, wdr + lor)
    w_ftr = _safe_div(wfta, wfga)
    w_pace = wfga - wor + wto + 0.475 * wfta
    l_pace = lfga - lor + lto + 0.475 * lfta

    win_side = pd.DataFrame(
        {
            "Season": win["Season"].astype(int),
            "TeamID": win["WTeamID"].astype(int),
            "DayNum": win["DayNum"].astype(int),
            "Win": np.ones(len(win), dtype=np.float32),
            "PtsFor": wscore,
            "PtsAgainst": lscore,
            "Margin": wscore - lscore,
            "eFG": w_efg,
            "Opp_eFG_Allowed": l_efg,
            "TOVRate": w_tov_rate,
            "Opp_TOV_Forced": l_tov_rate,
            "ORB": w_orb,
            "DRB": w_drb,
            "FTR": w_ftr,
            "PaceProxy": w_pace,
            "OppPaceProxy": l_pace,
            "Loc_H": win_home,
            "Loc_A": win_away,
            "Loc_N": win_neutral,
        }
    )

    loss = reg_detailed.copy()
    loss_home, loss_away, loss_neutral = _loc_onehot_team_perspective(loss["WLoc"], is_winner_side=False)

    lfgm = loss["LFGM"].to_numpy(dtype=np.float32)
    lfga = loss["LFGA"].to_numpy(dtype=np.float32)
    lfgm3 = loss["LFGM3"].to_numpy(dtype=np.float32)
    lfta = loss["LFTA"].to_numpy(dtype=np.float32)
    lto = loss["LTO"].to_numpy(dtype=np.float32)
    lor = loss["LOR"].to_numpy(dtype=np.float32)
    ldr = loss["LDR"].to_numpy(dtype=np.float32)
    lscore = loss["LScore"].to_numpy(dtype=np.float32)

    wfgm = loss["WFGM"].to_numpy(dtype=np.float32)
    wfga = loss["WFGA"].to_numpy(dtype=np.float32)
    wfgm3 = loss["WFGM3"].to_numpy(dtype=np.float32)
    wfta = loss["WFTA"].to_numpy(dtype=np.float32)
    wto = loss["WTO"].to_numpy(dtype=np.float32)
    wor = loss["WOR"].to_numpy(dtype=np.float32)
    wdr = loss["WDR"].to_numpy(dtype=np.float32)
    wscore = loss["WScore"].to_numpy(dtype=np.float32)

    l_efg = _safe_div(lfgm + 0.5 * lfgm3, lfga)
    w_efg = _safe_div(wfgm + 0.5 * wfgm3, wfga)
    l_tov_rate = _safe_div(lto, lfga + 0.44 * lfta + lto)
    w_tov_rate = _safe_div(wto, wfga + 0.44 * wfta + wto)
    l_orb = _safe_div(lor, lor + wdr)
    l_drb = _safe_div(ldr, ldr + wor)
    l_ftr = _safe_div(lfta, lfga)
    l_pace = lfga - lor + lto + 0.475 * lfta
    w_pace = wfga - wor + wto + 0.475 * wfta

    loss_side = pd.DataFrame(
        {
            "Season": loss["Season"].astype(int),
            "TeamID": loss["LTeamID"].astype(int),
            "DayNum": loss["DayNum"].astype(int),
            "Win": np.zeros(len(loss), dtype=np.float32),
            "PtsFor": lscore,
            "PtsAgainst": wscore,
            "Margin": lscore - wscore,
            "eFG": l_efg,
            "Opp_eFG_Allowed": w_efg,
            "TOVRate": l_tov_rate,
            "Opp_TOV_Forced": w_tov_rate,
            "ORB": l_orb,
            "DRB": l_drb,
            "FTR": l_ftr,
            "PaceProxy": l_pace,
            "OppPaceProxy": w_pace,
            "Loc_H": loss_home,
            "Loc_A": loss_away,
            "Loc_N": loss_neutral,
        }
    )

    out = pd.concat([win_side, loss_side], ignore_index=True)
    out = out.sort_values(["Season", "TeamID", "DayNum"], kind="mergesort").reset_index(drop=True)
    return out


def build_game_sequences(
    reg_detailed: pd.DataFrame,
    gender: str,
) -> dict[tuple[int, int], np.ndarray]:
    """Returns {(Season, TeamID): array of shape (T, input_dim)} sorted by DayNum."""
    del gender  # Kept for consistent call signatures and logging call sites.

    rows = _team_game_rows_from_detailed(reg_detailed)
    feature_cols = [
        "PtsFor",
        "PtsAgainst",
        "Margin",
        "eFG",
        "Opp_eFG_Allowed",
        "TOVRate",
        "Opp_TOV_Forced",
        "ORB",
        "DRB",
        "FTR",
        "PaceProxy",
        "OppPaceProxy",
        "Loc_H",
        "Loc_A",
        "Loc_N",
    ]

    sequences: dict[tuple[int, int], np.ndarray] = {}
    for (season, team_id), group in rows.groupby(["Season", "TeamID"], sort=False):
        arr = group[feature_cols].to_numpy(dtype=np.float32)
        if arr.shape[1] != INPUT_DIM:
            raise RuntimeError(f"Temporal input_dim mismatch: expected {INPUT_DIM}, got {arr.shape[1]}")
        sequences[(int(season), int(team_id))] = arr
    return sequences


@dataclass
class _SeasonBatch:
    x: torch.Tensor
    y_seq: torch.Tensor
    lengths: torch.Tensor
    keys: list[tuple[int, int]]


class _TemporalNet(nn.Module):
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        if model_type.lower() != "gru":
            raise ValueError(f"Unknown model_type={model_type}")
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        final_hidden = h_n[-1]
        logits = self.head(final_hidden).squeeze(-1)
        return logits, final_hidden


def _build_training_batches(
    sequences: dict[tuple[int, int], np.ndarray],
    games: pd.DataFrame,
) -> tuple[dict[int, _SeasonBatch], dict[tuple[int, int], np.ndarray]]:
    rows = _team_game_rows_from_detailed(games)

    labels_by_key: dict[tuple[int, int], np.ndarray] = {}
    for (season, team_id), group in rows.groupby(["Season", "TeamID"], sort=False):
        labels_by_key[(int(season), int(team_id))] = group["Win"].to_numpy(dtype=np.float32)

    all_keys = sorted(set(sequences.keys()) | set(labels_by_key.keys()))
    season_keys: dict[int, list[tuple[int, int]]] = {}
    for key in all_keys:
        season_keys.setdefault(key[0], []).append(key)

    zero_embeddings: dict[tuple[int, int], np.ndarray] = {}
    season_batches: dict[int, _SeasonBatch] = {}

    for season, keys in season_keys.items():
        eligible: list[tuple[int, int]] = []
        lengths: list[int] = []

        for key in keys:
            seq = sequences.get(key)
            y = labels_by_key.get(key)
            if seq is None or y is None:
                zero_embeddings[key] = np.zeros(0, dtype=np.float32)
                continue

            n = int(min(len(seq), len(y)))
            if n < MIN_GAMES_FOR_EMBED:
                zero_embeddings[key] = np.zeros(0, dtype=np.float32)
                continue

            eligible.append(key)
            lengths.append(n)

        if not eligible:
            continue

        max_len = int(max(lengths))
        x = np.zeros((len(eligible), max_len, INPUT_DIM), dtype=np.float32)
        y_seq = np.zeros((len(eligible),), dtype=np.float32)

        for i, key in enumerate(eligible):
            seq = sequences[key]
            tgt = labels_by_key[key]
            n = int(min(len(seq), len(tgt)))
            x[i, :n, :] = seq[:n]
            y_seq[i] = float(np.mean(tgt[:n]) >= 0.5)

        season_batches[season] = _SeasonBatch(
            x=torch.from_numpy(x),
            y_seq=torch.from_numpy(y_seq),
            lengths=torch.tensor(lengths, dtype=torch.long),
            keys=eligible,
        )

    return season_batches, zero_embeddings


def train_temporal_model(
    sequences: dict[tuple[int, int], np.ndarray],
    games: pd.DataFrame,
    model_type: str,
    hidden_dim: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[tuple[int, int], np.ndarray]:
    """Returns {(Season, TeamID): embedding array of shape (hidden_dim,)}."""
    if model_type.lower() != "gru":
        raise ValueError("model_type must be 'gru'")

    season_batches, zero_emb = _build_training_batches(sequences, games)
    embeddings: dict[tuple[int, int], np.ndarray] = {
        key: np.zeros(hidden_dim, dtype=np.float32) for key in sequences.keys()
    }

    if not season_batches:
        return embeddings

    device_obj = torch.device(device)
    model = _TemporalNet(model_type=model_type, input_dim=INPUT_DIM, hidden_dim=hidden_dim, num_layers=1).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    batch_size = 256

    model.train()
    for _ in range(int(epochs)):
        for season in sorted(season_batches):
            batch = season_batches[season]
            n_samples = batch.x.shape[0]
            order = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                x = batch.x[idx].to(device_obj)
                y = batch.y_seq[idx].to(device_obj)
                lengths = batch.lengths[idx].to(device_obj)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x, lengths)
                loss = bce(logits, y)
                loss.backward()
                optimizer.step()

    model.eval()
    with torch.no_grad():
        for season in sorted(season_batches):
            batch = season_batches[season]
            n_samples = batch.x.shape[0]
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x = batch.x[start:end].to(device_obj)
                lengths = batch.lengths[start:end].to(device_obj)
                _, h = model(x, lengths)
                h_np = h.detach().cpu().numpy().astype(np.float32)
                for i, key in enumerate(batch.keys[start:end]):
                    embeddings[key] = h_np[i]

    for key in zero_emb:
        if key in embeddings:
            embeddings[key] = np.zeros(hidden_dim, dtype=np.float32)

    return embeddings


def _pair_features(z1: np.ndarray, z2: np.ndarray, prefix: str) -> dict[str, float]:
    eps = 1e-12
    diff = z1 - z2
    prod = z1 * z2
    n1 = float(np.linalg.norm(z1))
    n2 = float(np.linalg.norm(z2))
    cos = float(np.dot(z1, z2) / (max(n1 * n2, eps)))
    dist = float(np.linalg.norm(diff))

    return {
        f"{prefix}_CosSim": cos,
        f"{prefix}_Dist": dist,
        f"{prefix}_DiffMean": float(np.mean(diff)),
        f"{prefix}_DiffStd": float(np.std(diff)),
        f"{prefix}_ProdMean": float(np.mean(prod)),
        f"{prefix}_Norm_T1": n1,
        f"{prefix}_Norm_T2": n2,
        f"{prefix}_NormDiff": n1 - n2,
    }


# NOTE: if NCAA_PHASE7_RETRAIN=0 and cached temporal_features_*.csv exists,
# verify it does NOT contain LSTM_* columns. If it does, set NCAA_PHASE7_RETRAIN=1
# to regenerate with GRU-only features.
def extract_temporal_features(
    games: pd.DataFrame,
    gru_embeddings: dict,
) -> pd.DataFrame:
    """Returns matchup-level DataFrame with GRU_* columns."""
    out_rows: list[dict[str, float | int]] = []

    hidden_dim = 64
    if gru_embeddings:
        first = next(iter(gru_embeddings.values()))
        if isinstance(first, np.ndarray) and first.ndim == 1 and first.size > 0:
            hidden_dim = int(first.size)

    z0 = np.zeros(hidden_dim, dtype=np.float32)

    keys_df = games[["Season", "Team1", "Team2"]].copy()
    for row in keys_df.itertuples(index=False):
        season = int(row.Season)
        team1 = int(row.Team1)
        team2 = int(row.Team2)
        k1 = (season, team1)
        k2 = (season, team2)

        g1 = gru_embeddings.get(k1, z0)
        g2 = gru_embeddings.get(k2, z0)

        feat = {
            "Season": season,
            "Team1": team1,
            "Team2": team2,
        }
        feat.update(_pair_features(np.asarray(g1, dtype=np.float32), np.asarray(g2, dtype=np.float32), "GRU"))
        out_rows.append(feat)

    return pd.DataFrame(out_rows)
