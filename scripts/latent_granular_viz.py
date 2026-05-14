#!/usr/bin/env python3
"""
Latent-granular matching diagnostics + graph/spectral/local-PCA views (ISMIR 2025 latent granular resynthesis).

Given a *source codebook* (stacked grains from one or more .npy under a folder) and a *target* .npy,
computes cosine similarity matching, softmax entropy over τ, optional assignment heatmaps,
codebook kNN / spectral embedding trajectory, sliding local PCA anisotropy & subspace angle.

Outputs PNG/CSV under --out-dir with hyperparameters encoded in filenames.

Interactive 3D (Three.js): run scripts/build_latent_granular_3d_viewer.py with the same
source folder + target npy to get a manipulable scene (joint PCA of codebook + target).

Requires: numpy, scipy, scikit-learn, matplotlib (optional seaborn for heatmap style).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enc24k_embedding_utils import (
    GrainMode,
    collect_source_codebook,
    load_embedding,
    l2_normalize_rows,
    make_grains,
)

try:
    from scipy.linalg import subspace_angles
except ImportError:
    subspace_angles = None  # type: ignore[misc,assignment]

from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def _batch_cosine_sim(
    target_n: np.ndarray,  # (M, d) normalized
    code_n: np.ndarray,  # (N, d) normalized
    batch: int = 256,
) -> np.ndarray:
    m, n = target_n.shape[0], code_n.shape[0]
    out = np.empty((m, n), dtype=np.float32)
    for s in range(0, m, batch):
        e = min(s + batch, m)
        out[s:e] = target_n[s:e] @ code_n.T
    return out


def _softmax_rows(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    z = x.astype(np.float64) / tau
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _entropy_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = -np.sum(p * np.log(p + eps), axis=1)
    return s.astype(np.float32)


def _kl_row(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) for two distributions; returns mean over dimensions if 1d."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def run_all(
    codebook: np.ndarray,
    target: np.ndarray,
    taus: list[float],
    heatmap_bins: int,
    graph_k: int,
    spectral_max_codebook: int,
    spectral_seed: int,
    local_window: int,
    local_k_cb: int,
    rng: np.random.Generator,
    out_prefix: str,
    skip_plots: bool,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = l2_normalize_rows(codebook.astype(np.float32))
    Tg = l2_normalize_rows(target.astype(np.float32))
    m, n_cb = Tg.shape[0], C.shape[0]
    d = Tg.shape[1]

    sims = _batch_cosine_sim(Tg, C, batch=256)
    ar = np.arange(m)
    if n_cb == 1:
        idx0 = np.zeros(m, dtype=np.int64)
        top1 = sims[:, 0].astype(np.float32)
        top2 = np.full(m, -1.0, dtype=np.float32)
        margin = top1 - top2
    else:
        idx0 = np.argmax(sims, axis=1).astype(np.int64)
        top1 = sims[ar, idx0].astype(np.float32)
        sims_wo = sims.copy()
        sims_wo[ar, idx0] = -2.0
        top2 = sims_wo.max(axis=1).astype(np.float32)
        margin = top1 - top2

    summary: dict[str, Any] = {"M": m, "N_codebook": n_cb, "D": d}

    # Time series CSV
    ts_path = f"{out_prefix}_matching_timeseries.csv"
    with open(ts_path, "w", encoding="utf-8") as f:
        hdr = ["t", "argmax_idx", "cos_best", "cos_second", "cos_margin"]
        for tau in taus:
            hdr.append(f"entropy_tau_{tau:g}")
        f.write(",".join(hdr) + "\n")
        ent_mat: dict[float, np.ndarray] = {}
        for tau in taus:
            P = _softmax_rows(sims, tau)
            ent_mat[tau] = _entropy_rows(P)
        for t in range(m):
            parts = [
                str(t),
                str(int(idx0[t])),
                f"{float(top1[t]):.8f}",
                f"{float(top2[t]):.8f}",
                f"{float(margin[t]):.8f}",
            ]
            for tau in taus:
                parts.append(f"{float(ent_mat[tau][t]):.8f}")
            f.write(",".join(parts) + "\n")
    summary["timeseries_csv"] = ts_path

    # Softmax flow KL between successive times (use first tau as reference)
    if taus:
        tau0 = taus[0]
        P = _softmax_rows(sims, tau0)
        kl_path = f"{out_prefix}_softmax_kl_tau{tau0:g}.csv"
        with open(kl_path, "w", encoding="utf-8") as fk:
            fk.write("t,kl_fwd,kl_rev,sym\n")
            for t in range(m - 1):
                kf = _kl_row(P[t + 1], P[t])
                kr = _kl_row(P[t], P[t + 1])
                fk.write(f"{t},{kf:.8f},{kr:.8f},{0.5 * (kf + kr):.8f}\n")
        summary["softmax_kl_csv"] = kl_path

    if skip_plots:
        return summary

    # --- Heatmap: target time x binned codebook index ---
    pca1 = PCA(n_components=1, random_state=0).fit(C)
    c1 = pca1.transform(C).ravel()
    lo, hi = np.percentile(c1, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, heatmap_bins + 1)
    bin_idx = np.clip(np.digitize(c1, edges) - 1, 0, heatmap_bins - 1)
    # Aggregate argmax hits
    H = np.zeros((m, heatmap_bins), dtype=np.float32)
    for t in range(m):
        b = bin_idx[int(idx0[t])]
        H[t, b] += 1.0
    # optional: softmax mass per bin for tau0
    if taus:
        P = _softmax_rows(sims, taus[0])
        Hmass = np.zeros_like(H)
        for t in range(m):
            for j in range(n_cb):
                Hmass[t, bin_idx[j]] += P[t, j]
        hplot = Hmass
    else:
        hplot = H

    fig, ax = plt.subplots(figsize=(10, max(3, m * 0.004)), dpi=140)
    ax.imshow(hplot, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
    ax.set_xlabel(f"codebook bin (PCA1 of codebook, {heatmap_bins} bins)")
    ax.set_ylabel("target grain index t")
    tau_note = f" τ={taus[0]}" if taus else " (argmax mass)"
    ax.set_title(f"Latent-granular assignment heatmap{tau_note}")
    fig.tight_layout()
    hp = f"{out_prefix}_heatmap_bins{heatmap_bins}.png"
    fig.savefig(hp, bbox_inches="tight")
    plt.close(fig)
    summary["heatmap_png"] = hp

    # Index plot
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=140)
    ax.plot(np.arange(m), idx0, lw=0.8, color="#1565c0")
    ax.set_xlabel("target t")
    ax.set_ylabel("argmax codebook grain index")
    ax.set_title("Chosen codebook index vs time (argmax cosine)")
    fig.tight_layout()
    ip = f"{out_prefix}_argmax_index.png"
    fig.savefig(ip, bbox_inches="tight")
    plt.close(fig)
    summary["index_png"] = ip

    # Margin + entropy panel
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=140, sharex=True)
    axes[0].plot(margin, lw=0.9, color="#2e7d32")
    axes[0].set_ylabel("margin (top1 - top2 cosine)")
    axes[0].set_title("Matching diagnostics")
    for tau in taus[: min(4, len(taus))]:
        P = _softmax_rows(sims, tau)
        axes[1].plot(_entropy_rows(P), lw=0.8, label=f"τ={tau:g}")
    axes[1].set_ylabel("softmax entropy")
    axes[1].set_xlabel("t")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    dp = f"{out_prefix}_margin_entropy.png"
    fig.savefig(dp, bbox_inches="tight")
    plt.close(fig)
    summary["diag_png"] = dp

    # --- Spectral embedding on subsampled codebook + target trajectory via NN indices ---
    n_sub = min(spectral_max_codebook, n_cb)
    if n_sub < 8:
        summary["spectral_skip"] = "codebook too small"
    else:
        sub_idx = rng.choice(n_cb, size=n_sub, replace=False)
        C_sub = C[sub_idx]
        try:
            embedder = SpectralEmbedding(
                n_components=2,
                random_state=spectral_seed,
                affinity="nearest_neighbors",
                n_neighbors=min(graph_k, n_sub - 1),
            )
            XY_sub = embedder.fit_transform(C_sub)
        except Exception as e:  # noqa: BLE001
            summary["spectral_error"] = str(e)
            XY_sub = PCA(n_components=2, random_state=spectral_seed).fit_transform(C_sub)

        # Trajectory: for each target grain, use argmax codebook index -> if in inv, else nearest in C_sub by cosine
        nn_sub = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn_sub.fit(C_sub)
        traj = np.zeros((m, 2), dtype=np.float64)
        _, sub_nn = nn_sub.kneighbors(Tg)
        for t in range(m):
            j = int(sub_nn[t, 0])
            traj[t] = XY_sub[j]

        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        ax.scatter(XY_sub[:, 0], XY_sub[:, 1], s=3, c="#bdbdbd", alpha=0.35, linewidths=0)
        ax.plot(traj[:, 0], traj[:, 1], color="#c41e3a", lw=1.2, alpha=0.9, label="target trajectory (NN in sub-codebook)")
        ax.scatter(traj[0, 0], traj[0, 1], s=36, c="green", marker="o", zorder=5, label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], s=36, c="black", marker="s", zorder=5, label="end")
        ax.set_title(
            f"Codebook spectral 2D (subsample n={n_sub}, affinity knn k={graph_k}); "
            f"target mapped by nearest grain in subsample"
        )
        ax.legend(loc="best", fontsize=8)
        sp = f"{out_prefix}_spectral_sub{n_sub}_k{graph_k}.png"
        fig.tight_layout()
        fig.savefig(sp, bbox_inches="tight")
        plt.close(fig)
        summary["spectral_png"] = sp

    # --- Sparse kNN graph stats + walk illustration (t vs subspace NN index) ---
    knn_graph = kneighbors_graph(
        C,
        n_neighbors=min(graph_k, max(1, n_cb - 1)),
        mode="connectivity",
        include_self=False,
        metric="cosine",
    )
    degrees = np.array(knn_graph.sum(axis=1)).ravel()
    summary["graph_mean_degree"] = float(np.mean(degrees))

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=140)
    sub_nn_full = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="cosine").fit(C)
    walk_j, _ = sub_nn_full.kneighbors(Tg)
    ax.plot(walk_j.ravel(), lw=0.7, color="#6a1b9a")
    ax.set_xlabel("t")
    ax.set_ylabel("nearest codebook index (full book, cosine)")
    ax.set_title("Walk: nearest codebook grain per target step (cosine NN, not necessarily argmax)")
    wp = f"{out_prefix}_nn_walk_k{graph_k}.png"
    fig.tight_layout()
    fig.savefig(wp, bbox_inches="tight")
    plt.close(fig)
    summary["walk_png"] = wp

    # --- Local PCA: sliding window on target; compare subspace to kNN codebook patch ---
    if local_window < 2 or local_k_cb < 4:
        summary["local_pca_skip"] = "bad window/k"
    else:
        half = local_window // 2
        aniso: list[float] = []
        angles: list[float] = []
        nn_cb_all = NearestNeighbors(n_neighbors=min(local_k_cb, n_cb), metric="cosine", algorithm="auto")
        nn_cb_all.fit(C)
        for t in range(m):
            s = max(0, t - half)
            e = min(m, t + half + 1)
            W = Tg[s:e]
            if W.shape[0] < 2:
                aniso.append(float("nan"))
                angles.append(float("nan"))
                continue
            pW = PCA(n_components=min(3, W.shape[0], W.shape[1]))
            pW.fit(W)
            ev = pW.explained_variance_
            aniso.append(float((ev[0] + 1e-12) / (ev[1] + 1e-12)) if ev.size > 1 else float("nan"))
            z = Tg[t : t + 1]
            _, ind = nn_cb_all.kneighbors(z)
            patch = C[ind.ravel()]
            pC = PCA(n_components=min(3, patch.shape[0], patch.shape[1]))
            pC.fit(patch)
            if subspace_angles is not None and pW.components_.shape[0] >= 2 and pC.components_.shape[0] >= 2:
                A = pW.components_[:2].T
                B = pC.components_[:2].T
                ang = subspace_angles(A, B)
                angles.append(float(np.max(ang)))
            else:
                angles.append(float("nan"))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=140, sharex=True)
        axes[0].plot(aniso, lw=0.9, color="#ef6c00")
        axes[0].set_ylabel("anisotropy λ₁/λ₂ (target window)")
        axes[1].plot(angles, lw=0.9, color="#00838f")
        axes[1].set_ylabel("subspace angle (rad)")
        axes[1].set_xlabel("t")
        axes[0].set_title(
            f"Local PCA: window={local_window}, codebook patch k={local_k_cb} (scipy subspace_angles)"
        )
        fig.tight_layout()
        lp = f"{out_prefix}_local_pca_w{local_window}_kpatch{local_k_cb}.png"
        fig.savefig(lp, bbox_inches="tight")
        plt.close(fig)
        summary["local_pca_png"] = lp

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--enc-root", default="data/enc24k")
    p.add_argument("--source-folder", required=True, help="e.g. instrument_samples")
    p.add_argument("--source-glob", default="*.npy")
    p.add_argument("--max-source-files", type=int, default=None)
    p.add_argument("--target-npy", required=True, help="Path to target .npy")
    p.add_argument("--out-dir", default="results/latent_granular")
    p.add_argument("--tag", default="", help="Filename tag; default derived from folders")
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--grain-mode", choices=("mean", "concat"), default="mean")
    p.add_argument(
        "--assume-layout",
        choices=("row_is_time", "col_is_time"),
        default="row_is_time",
    )
    p.add_argument("--taus", default="0.05,0.1,0.2,0.5", help="Comma-separated softmax temperatures")
    p.add_argument("--heatmap-bins", type=int, default=64)
    p.add_argument("--graph-k", type=int, default=12)
    p.add_argument("--spectral-max-codebook", type=int, default=4000)
    p.add_argument("--local-window", type=int, default=16)
    p.add_argument("--local-k-cb", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-plots", action="store_true", help="Only write CSV diagnostics")
    args = p.parse_args()

    assume_row = args.assume_layout == "row_is_time"
    stride = args.stride if args.stride is not None else args.window_size
    rng = np.random.default_rng(args.seed)
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    codebook, src_stems = collect_source_codebook(
        args.enc_root,
        args.source_folder,
        glob_pattern=args.source_glob,
        max_files=args.max_source_files,
        window_size=args.window_size,
        stride=stride,
        grain_mode=args.grain_mode,  # type: ignore[arg-type]
        assume_row_is_time=assume_row,
    )
    tx = load_embedding(args.target_npy, assume_row_is_time=assume_row)
    if args.window_size == 1 and stride == 1:
        target = tx
    else:
        target = make_grains(tx, args.window_size, stride, args.grain_mode)  # type: ignore[arg-type]

    os.makedirs(args.out_dir, exist_ok=True)
    tgt_name = os.path.splitext(os.path.basename(args.target_npy))[0]
    tag = args.tag or f"{args.source_folder}_to_{tgt_name}_w{args.window_size}_s{stride}_{args.grain_mode}"
    out_prefix = os.path.join(args.out_dir, tag)

    summ = run_all(
        codebook=codebook,
        target=target,
        taus=taus,
        heatmap_bins=args.heatmap_bins,
        graph_k=args.graph_k,
        spectral_max_codebook=args.spectral_max_codebook,
        spectral_seed=args.seed,
        local_window=args.local_window,
        local_k_cb=args.local_k_cb,
        rng=rng,
        out_prefix=out_prefix,
        skip_plots=args.skip_plots,
    )
    print("Done.", summ)


if __name__ == "__main__":
    main()
