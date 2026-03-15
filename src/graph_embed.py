"""Phase 6: R-GCN with SSL for NCAA graph embeddings and matchup features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import RGCNConv


REL_HOME_WIN = 0
REL_HOME_LOSS = 1
REL_AWAY_WIN = 2
REL_AWAY_LOSS = 3
REL_NEUT_WIN = 4
REL_NEUT_LOSS = 5
NUM_RELATIONS = 6

WL_CLASS_LOSS = 0
WL_CLASS_WIN = 1


class RGCN(nn.Module):
    """Relational GCN for location-aware win/loss edge types."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_relations: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(RGCNConv(in_c, out_c, num_relations, aggr="mean"))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_c))
                self.acts.append(nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.acts):
                x = self.norms[i](x)
                x = self.acts[i](x)
        return x


class SSLTaskHead(nn.Module):
    """Self-supervised edge heads: win/loss and margin buckets."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.win_loss_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),
        )
        self.margin_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 5),
        )

    def forward(self, src_z: torch.Tensor, dst_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([src_z, dst_z], dim=1)
        return self.win_loss_head(concat), self.margin_head(concat)


def _margin_to_bucket(margin: float) -> int:
    m = abs(float(margin))
    if m < 5:
        return 0
    if m < 10:
        return 1
    if m < 15:
        return 2
    if m < 20:
        return 3
    return 4


def _build_graph_from_games(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    game_locations: pd.DataFrame | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Build directed graph with explicit win/loss relation types."""
    games_sorted = games.drop_duplicates(subset=["Season", "Team1", "Team2"]).sort_values(["Season", "Team1", "Team2"])

    nodes = pd.concat(
        [
            games_sorted[["Season", "Team1"]].rename(columns={"Team1": "TeamID"}),
            games_sorted[["Season", "Team2"]].rename(columns={"Team2": "TeamID"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    nodes = nodes.sort_values(["Season", "TeamID"]).reset_index(drop=True)

    node_to_idx: Dict[tuple[int, int], int] = {
        (int(row.Season), int(row.TeamID)): i for i, row in enumerate(nodes.itertuples(index=False))
    }
    idx_to_node: Dict[int, tuple[int, int]] = {v: k for k, v in node_to_idx.items()}

    tf = team_features.copy()
    numeric_cols = [c for c in tf.columns if c not in ["Season", "TeamID"]]
    tf_index = tf.set_index(["Season", "TeamID"])

    x_rows: list[np.ndarray] = []
    for i in range(len(nodes)):
        season, team_id = idx_to_node[i]
        key = (season, team_id)
        if key in tf_index.index:
            vals = tf_index.loc[key, numeric_cols]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[0]
            x_rows.append(np.asarray(vals, dtype=np.float32))
        else:
            x_rows.append(np.zeros(len(numeric_cols), dtype=np.float32))

    x = torch.tensor(np.asarray(x_rows), dtype=torch.float32)

    src_list: list[int] = []
    dst_list: list[int] = []
    rel_list: list[int] = []
    margin_list: list[float] = []

    loc_lookup: Dict[tuple[int, int, int], str] = {}
    if game_locations is not None and not game_locations.empty:
        req_cols = ["Season", "Team1", "Team2", "WLoc"]
        if all(c in game_locations.columns for c in req_cols):
            gl = game_locations[req_cols].copy()
            gl["WLoc"] = gl["WLoc"].fillna("N").astype(str).str.upper()
            gl.loc[~gl["WLoc"].isin(["H", "A", "N"]), "WLoc"] = "N"
            # Keep dedup policy aligned with games_sorted (default keep='first')
            # so location rows map to the same canonical game row used for y_true/margin.
            gl = gl.drop_duplicates(subset=["Season", "Team1", "Team2"], keep="first")
            loc_lookup = {
                (int(r.Season), int(r.Team1), int(r.Team2)): str(r.WLoc)
                for r in gl.itertuples(index=False)
            }

    for row in games_sorted.itertuples(index=False):
        season = int(row.Season)
        t1 = int(row.Team1)
        t2 = int(row.Team2)
        y_true = int(row.y_true)
        margin = float(getattr(row, "Margin", 0.0))

        i1 = node_to_idx.get((season, t1))
        i2 = node_to_idx.get((season, t2))
        if i1 is None or i2 is None:
            continue

        # Canonical y_true: 1 => Team1 won, else Team2 won.
        winner_i, loser_i = (i1, i2) if y_true == 1 else (i2, i1)

        # WLoc is winner-relative location from compact results.
        loc = loc_lookup.get((season, t1, t2), "N")
        if loc == "H":
            win_rel, loss_rel = REL_HOME_WIN, REL_HOME_LOSS
        elif loc == "A":
            win_rel, loss_rel = REL_AWAY_WIN, REL_AWAY_LOSS
        else:
            win_rel, loss_rel = REL_NEUT_WIN, REL_NEUT_LOSS

        # winner -> loser (win relation)
        src_list.append(winner_i)
        dst_list.append(loser_i)
        rel_list.append(win_rel)
        margin_list.append(abs(margin))

        # loser -> winner (loss relation)
        src_list.append(loser_i)
        dst_list.append(winner_i)
        rel_list.append(loss_rel)
        margin_list.append(abs(margin))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(rel_list, dtype=torch.long)
        edge_margin = torch.tensor(np.asarray(margin_list, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_margin = torch.empty((0, 1), dtype=torch.float32)

    return x, edge_index, edge_type, edge_margin, {
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
    }


def train_graph_embedding(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    game_locations: pd.DataFrame | None = None,
    gender: str = "m",
    embedding_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 30,
    lr: float = 0.001,
    val_season_max: int = 2019,
    device: str = "cpu",
    embeddings_path: Path | None = None,
) -> Tuple[Dict[tuple[int, int], np.ndarray], np.ndarray, pd.DataFrame]:
    """Train R-GCN + SSL and return embeddings plus graph-derived edge win probabilities."""
    del val_season_max  # Kept for backward-compatible call sites.

    print(f"[GRAPH] gender={gender.upper()} building graph...", flush=True)
    x, edge_index, edge_type, edge_margin, meta = _build_graph_from_games(
        games,
        team_features,
        game_locations=game_locations,
    )
    print(f"[GRAPH] gender={gender.upper()} nodes={x.shape[0]} edges={edge_index.shape[1]}", flush=True)

    if edge_index.shape[1] < 10:
        print(f"[GRAPH] WARNING: insufficient edges ({edge_index.shape[1]}), skipping graph embedding", flush=True)
        return {}, np.zeros((x.shape[0], embedding_dim), dtype=np.float32), pd.DataFrame()

    device_obj = torch.device(device)
    x = x.to(device_obj)
    edge_index = edge_index.to(device_obj)
    edge_type = edge_type.to(device_obj)
    edge_margin = edge_margin.to(device_obj)

    model = RGCN(
        in_channels=x.shape[1],
        hidden_channels=embedding_dim,
        out_channels=embedding_dim,
        num_relations=NUM_RELATIONS,
        num_layers=num_layers,
    ).to(device_obj)
    head = SSLTaskHead(embedding_dim).to(device_obj)

    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    loss_wl = nn.CrossEntropyLoss()
    loss_margin = nn.CrossEntropyLoss()

    print(f"[GRAPH] gender={gender.upper()} training R-GCN + SSL...", flush=True)
    for epoch in range(epochs):
        model.train()
        head.train()

        z = model(x, edge_index, edge_type)
        src_z = z[edge_index[0]]
        dst_z = z[edge_index[1]]
        wl_logits, margin_logits = head(src_z, dst_z)

        margin_targets = torch.tensor(
            np.asarray([_margin_to_bucket(v) for v in edge_margin[:, 0].detach().cpu().numpy()], dtype=np.int64),
            dtype=torch.long,
            device=device_obj,
        )
        wl_targets = torch.isin(
            edge_type,
            torch.tensor([REL_HOME_WIN, REL_AWAY_WIN, REL_NEUT_WIN], device=device_obj),
        ).long()

        loss_wl_val = loss_wl(wl_logits, wl_targets)
        loss_margin_val = loss_margin(margin_logits, margin_targets)
        loss = loss_wl_val + 0.5 * loss_margin_val

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[GRAPH] gender={gender.upper()} epoch={epoch + 1}/{epochs} "
                f"loss={loss.item():.4f} "
                f"loss_wl={loss_wl_val.item():.4f} "
                f"loss_margin={loss_margin_val.item():.4f}",
                flush=True,
            )

    model.eval()
    head.eval()
    with torch.no_grad():
        z_final = model(x, edge_index, edge_type)
        win_logits, _ = head(z_final[edge_index[0]], z_final[edge_index[1]])
        win_probs = torch.softmax(win_logits, dim=1)[:, WL_CLASS_WIN].detach().cpu().numpy()

    z_np = z_final.detach().cpu().numpy()
    embeddings_dict: Dict[tuple[int, int], np.ndarray] = {
        meta["idx_to_node"][i]: z_np[i] for i in range(z_np.shape[0])
    }
    if embeddings_path is not None:
        _save_embeddings(embeddings_dict, embeddings_path)

    # Build per-matchup GraphWinProb from directed edge scores.
    dir_scores = {}
    ei_cpu = edge_index.detach().cpu().numpy()
    for i in range(ei_cpu.shape[1]):
        src_i = int(ei_cpu[0, i])
        dst_i = int(ei_cpu[1, i])
        src_key = meta["idx_to_node"][src_i]
        dst_key = meta["idx_to_node"][dst_i]
        dir_scores[(src_key[0], src_key[1], dst_key[1])] = float(win_probs[i])

    rows = []
    for row in games.itertuples(index=False):
        season = int(row.Season)
        t1 = int(row.Team1)
        t2 = int(row.Team2)
        rows.append(
            {
                "Season": season,
                "Team1": t1,
                "Team2": t2,
                "GraphWinProb": dir_scores.get((season, t1, t2), 0.5),
            }
        )
    graph_winprob_df = pd.DataFrame(rows)
    if not graph_winprob_df.empty:
        graph_winprob_df = (
            graph_winprob_df.groupby(["Season", "Team1", "Team2"], as_index=False)["GraphWinProb"]
            .mean()
        )

    print(f"[GRAPH] gender={gender.upper()} completed. Generated {len(embeddings_dict)} embeddings.", flush=True)
    return embeddings_dict, z_np, graph_winprob_df


def _save_embeddings(
    embeddings_dict: Dict[tuple[int, int], np.ndarray],
    path: Path,
) -> None:
    rows = [
        {"Season": key[0], "TeamID": key[1], **{f"emb_{i}": float(vec[i]) for i in range(len(vec))}}
        for key, vec in embeddings_dict.items()
    ]
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def _load_embeddings(path: Path) -> Dict[tuple[int, int], np.ndarray]:
    df = pd.read_parquet(path)
    if df.empty:
        return {}

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_")[1]))
    out: Dict[tuple[int, int], np.ndarray] = {}
    for row in df.itertuples(index=False):
        season = int(getattr(row, "Season"))
        team_id = int(getattr(row, "TeamID"))
        vec = np.asarray([float(getattr(row, c)) for c in emb_cols], dtype=np.float32)
        out[(season, team_id)] = vec
    return out


def extract_graph_features(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    embeddings_dict: Dict[tuple[int, int], np.ndarray],
    gender: str = "m",
    n_clusters: int = 8,
    n_neighbors: int = 5,
    graph_winprob: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract matchup geometry, strength, archetype, and neighborhood features."""
    del team_features

    if not embeddings_dict:
        print(f"[FEATURES] gender={gender.upper()} no embeddings, returning empty frame", flush=True)
        return pd.DataFrame()

    print(f"[FEATURES] gender={gender.upper()} extracting graph features...", flush=True)
    node_keys = list(embeddings_dict.keys())
    embed_array = np.asarray([embeddings_dict[k] for k in node_keys], dtype=np.float32)

    n_clusters_eff = max(2, min(n_clusters, embed_array.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embed_array)
    cluster_dict = {node_keys[i]: int(cluster_labels[i]) for i in range(len(node_keys))}

    if embed_array.shape[0] > 1:
        n_neighbors_eff = max(1, min(n_neighbors, embed_array.shape[0] - 1))
        knn = NearestNeighbors(n_neighbors=n_neighbors_eff, algorithm="auto")
        knn.fit(embed_array)
        _, indices = knn.kneighbors(embed_array)
        neighbor_strength = np.linalg.norm(np.asarray([embed_array[idx].mean(axis=0) for idx in indices]), axis=1)
    else:
        neighbor_strength = np.ones((embed_array.shape[0],), dtype=np.float32)
    neighbor_dict = {node_keys[i]: float(neighbor_strength[i]) for i in range(len(node_keys))}

    wp = None
    if graph_winprob is not None and not graph_winprob.empty:
        wp = graph_winprob.set_index(["Season", "Team1", "Team2"])["GraphWinProb"].to_dict()

    rows = []
    for row in games.itertuples(index=False):
        season = int(row.Season)
        t1 = int(row.Team1)
        t2 = int(row.Team2)
        n1 = (season, t1)
        n2 = (season, t2)
        if n1 not in embeddings_dict or n2 not in embeddings_dict:
            continue

        z1 = embeddings_dict[n1]
        z2 = embeddings_dict[n2]

        cos_sim = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8))
        dist = float(np.linalg.norm(z1 - z2))
        dot_prod = float(np.dot(z1, z2))
        z_diff = z1 - z2
        z_prod = z1 * z2

        norm1 = float(np.linalg.norm(z1))
        norm2 = float(np.linalg.norm(z2))
        c1 = int(cluster_dict.get(n1, -1))
        c2 = int(cluster_dict.get(n2, -1))
        ns1 = float(neighbor_dict.get(n1, 0.0))
        ns2 = float(neighbor_dict.get(n2, 0.0))

        rows.append(
            {
                "Season": season,
                "Team1": t1,
                "Team2": t2,
                "EmbedCosSim": cos_sim,
                "EmbedDist": dist,
                "EmbedDot": dot_prod,
                "EmbedDiffMean": float(np.mean(z_diff)),
                "EmbedDiffStd": float(np.std(z_diff)),
                "EmbedProdMean": float(np.mean(z_prod)),
                "EmbedProdStd": float(np.std(z_prod)),
                "EmbedNorm_T1": norm1,
                "EmbedNorm_T2": norm2,
                "EmbedNormDiff": norm1 - norm2,
                "EmbedNormRatio": (norm1 + 1e-8) / (norm2 + 1e-8),
                "Cluster_T1": c1,
                "Cluster_T2": c2,
                "ClusterMatch": int(c1 == c2),
                "ArchetypePair": f"{min(c1, c2)}_vs_{max(c1, c2)}",
                "NeighborStrength_T1": ns1,
                "NeighborStrength_T2": ns2,
                "NeighborStrengthDiff": ns1 - ns2,
                "GraphWinProb": float(wp.get((season, t1, t2), 0.5)) if wp is not None else 0.5,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        key_cols = ["Season", "Team1", "Team2"]
        numeric_cols = [c for c in out.columns if c not in key_cols and pd.api.types.is_numeric_dtype(out[c])]
        text_cols = [c for c in out.columns if c not in key_cols and c not in numeric_cols]
        agg: dict[str, str] = {c: "mean" for c in numeric_cols}
        agg.update({c: "first" for c in text_cols})
        out = out.groupby(key_cols, as_index=False).agg(agg)
    print(f"[FEATURES] gender={gender.upper()} extracted {len(out)} game features", flush=True)
    return out


def integrate_graph_features(train_df: pd.DataFrame, graph_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join graph features on matchup keys."""
    if graph_features.empty:
        return train_df
    return train_df.merge(graph_features, on=["Season", "Team1", "Team2"], how="left")
