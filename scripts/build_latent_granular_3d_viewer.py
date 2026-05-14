#!/usr/bin/env python3
"""
Interactive 3D viewer (Three.js) for latent-granular geometry in a *shared* 3D chart.

The viewer HTML loads **per-pair JSON bundles** from a `bundles/` directory next to the HTML.
You choose **codebook source folder** + **target sound** in dropdowns; each combination must
have been pre-exported (see below).

  • Subsample the source codebook for display (many grains).
  • Fit PCA(3) on stacked rows [codebook_subsample ; target grains] so both live in one space.
  • Draw target and argmax paths as **glass-like tubes** (moving internal highlight) plus optional
    **alpha-shape tight hull** meshes embedded in each bundle (needs **SciPy** at export).
  • `--no-tight-hulls` skips mesh export; `--hull-alpha-percentile` tunes hull tightness (lower ≈ tighter).

Serve over HTTP, e.g.:
  cd results/latent_granular && python3 -m http.server 8766
  → http://localhost:8766/latent_granular_3d_viewer.html

Typical workflow
----------------
1) Write viewer + manifest (scans data/enc24k for folders + .npy stems):

     python3 scripts/build_latent_granular_3d_viewer.py \\
       --enc-root data/enc24k --out results/latent_granular/latent_granular_3d_viewer.html

   Same as: pass --manifest-only if you only want to refresh manifest + HTML.

2) Export each (source, target) pair you care about (creates bundles/*.json):

     python3 scripts/build_latent_granular_3d_viewer.py \\
       --source-folder instrument_samples --target-npy data/enc24k/soundscapes/149916_2687794.npy \\
       --out results/latent_granular/latent_granular_3d_viewer.html

3) Or precompute **every** codebook-folder × target combination (can take several minutes):

     python3 scripts/build_latent_granular_3d_viewer.py --build-all-bundles \\
       --out results/latent_granular/latent_granular_3d_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enc24k_embedding_utils import (
    GrainMode,
    collect_source_codebook,
    grain_fingerprint,
    load_embedding,
    l2_normalize_rows,
    make_grains,
)


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = 0.5 * (lo + hi)
    span = np.max(hi - lo)
    if span <= 1e-12:
        span = 1.0
    return (xyz - c) * (2.2 / span)


def _normalize_xyz_center_span(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Center / max-edge span used to align subset projections with a fixed corpus chart."""
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if span <= 1e-12:
        span = 1.0
    return c.astype(np.float64), span


def _apply_normalize_xyz(xyz: np.ndarray, center: np.ndarray, span: float) -> np.ndarray:
    span = float(span) if float(span) > 1e-12 else 1.0
    return ((np.asarray(xyz, dtype=np.float64) - np.asarray(center, dtype=np.float64)) * (2.2 / span)).astype(
        np.float32
    )


def corpus_cloud_json_path_from_pca_npz(npz_path: str) -> str:
    if npz_path.endswith("_pca.npz"):
        return npz_path[: -len("_pca.npz")] + "_corpus_cloud.json"
    stem, _ = os.path.splitext(npz_path)
    return stem + "_corpus_cloud.json"


def install_global_corpus_cloud(npz_path: str | None, bundles_dir: str) -> str | None:
    """Copy *_corpus_cloud.json next to the PCA npz into bundles/. Returns manifest-relative URL or None."""
    if not npz_path or not os.path.isfile(npz_path):
        return None
    src = corpus_cloud_json_path_from_pca_npz(npz_path)
    if not os.path.isfile(src):
        return None
    name = os.path.basename(src)
    dst = os.path.join(bundles_dir, name)
    shutil.copy2(src, dst)
    return "bundles/" + name


def resolve_global_corpus_cloud_manifest_url(g_npz: str | None, bundles_dir: str) -> str | None:
    """Install from npz path, or keep existing manifest entry if the JSON file is still in bundles/."""
    fresh = install_global_corpus_cloud(g_npz, bundles_dir)
    if fresh:
        return fresh
    mp = os.path.join(bundles_dir, "manifest.json")
    prev_m: dict[str, Any] = {}
    if os.path.isfile(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                prev_m = json.load(f)
        except (OSError, json.JSONDecodeError):
            prev_m = {}
    prev_url = prev_m.get("global_corpus_cloud")
    if prev_url and isinstance(prev_url, str):
        fname = prev_url
        if prev_url.startswith("bundles/"):
            fname = prev_url[len("bundles/") :]
        fname = fname.lstrip("/")
        if fname and os.path.isfile(os.path.join(bundles_dir, fname)):
            return prev_url if prev_url.startswith("bundles/") else "bundles/" + fname
    candidates = sorted(f for f in os.listdir(bundles_dir) if f.endswith("_corpus_cloud.json"))
    if len(candidates) == 1:
        return "bundles/" + candidates[0]
    return None


def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _b64_u32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint32).tobytes()).decode("ascii")


def _pack_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    return {
        "vcount": int(vertices.shape[0]),
        "fcount": int(faces.shape[0]),
        "pos_b64": _b64_f32(vertices.astype(np.float32).reshape(-1)),
        "idx_b64": _b64_u32(faces.astype(np.uint32).reshape(-1)),
    }


def _subsample_points(pts: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(pts.shape[0])
    if n <= max_n:
        return np.asarray(pts, dtype=np.float64)
    pick = rng.choice(n, size=max_n, replace=False)
    return np.asarray(pts[pick], dtype=np.float64)


def _tet_circumradius_sq(pts: np.ndarray, ix: np.ndarray) -> float:
    p0 = pts[ix[0]]
    t1 = pts[ix[1]] - p0
    t2 = pts[ix[2]] - p0
    t3 = pts[ix[3]] - p0
    mat = 2.0 * np.stack([t1, t2, t3], axis=0)
    rhs = np.array(
        [float(np.dot(t1, t1)), float(np.dot(t2, t2)), float(np.dot(t3, t3))],
        dtype=np.float64,
    )
    if abs(np.linalg.det(mat)) < 1e-18:
        return float("inf")
    c = np.linalg.solve(mat, rhs)
    return float(np.dot(c, c))


def _convex_hull_mesh(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    from scipy.spatial import ConvexHull, QhullError

    pts64 = np.asarray(pts, dtype=np.float64)
    try:
        hull = ConvexHull(pts64)
    except QhullError:
        return None
    faces = hull.simplices.astype(np.int32)
    uix = np.unique(faces.reshape(-1))
    idx_map = np.full(pts64.shape[0], -1, dtype=np.int32)
    for new_i, old_i in enumerate(uix):
        idx_map[old_i] = new_i
    verts = np.asarray(pts, dtype=np.float32)[uix]
    faces_r = idx_map[faces]
    if np.any(faces_r < 0):
        return None
    return verts, faces_r


def _alpha_shape_boundary(pts: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray] | None:
    from scipy.spatial import Delaunay, QhullError

    pts64 = np.asarray(pts, dtype=np.float64)
    n = pts64.shape[0]
    if n < 4:
        return None
    try:
        tri = Delaunay(pts64)
    except QhullError:
        return _convex_hull_mesh(pts)

    simplices = tri.simplices
    radii = np.empty(len(simplices), dtype=np.float64)
    for i, s in enumerate(simplices):
        r2 = _tet_circumradius_sq(pts64, s)
        if math.isfinite(r2) and r2 > 1e-20:
            radii[i] = math.sqrt(r2)
        else:
            radii[i] = float("nan")

    valid = np.isfinite(radii) & (radii > 1e-12)
    if not np.any(valid):
        return _convex_hull_mesh(pts)
    rfin = radii[valid]
    p = float(np.clip(percentile, 5.0, 95.0))
    alpha_r = float(np.percentile(rfin, p))
    kept = simplices[radii <= alpha_r * (1.0 + 1e-6)]
    if len(kept) == 0:
        alpha_r = float(np.min(rfin) + (np.median(rfin) - np.min(rfin)) * 0.85)
        kept = simplices[radii <= alpha_r * (1.0 + 1e-6)]
    if len(kept) == 0:
        return _convex_hull_mesh(pts)

    cnt: dict[tuple[int, ...], int] = {}
    for tet in kept:
        fabc = tuple(int(x) for x in sorted([tet[0], tet[1], tet[2]]))
        fabd = tuple(int(x) for x in sorted([tet[0], tet[1], tet[3]]))
        facd = tuple(int(x) for x in sorted([tet[0], tet[2], tet[3]]))
        fbcd = tuple(int(x) for x in sorted([tet[1], tet[2], tet[3]]))
        for tri in (fabc, fabd, facd, fbcd):
            cnt[tri] = cnt.get(tri, 0) + 1
    btris = [list(t) for t, c in cnt.items() if c == 1]
    if len(btris) == 0:
        return _convex_hull_mesh(pts)

    faces = np.asarray(btris, dtype=np.int32)
    uix = np.unique(faces.reshape(-1))
    idx_map = np.full(pts64.shape[0], -1, dtype=np.int32)
    for new_i, old_i in enumerate(uix):
        idx_map[old_i] = new_i
    verts = np.asarray(pts, dtype=np.float32)[uix]
    faces_r = idx_map[faces]
    if np.any(faces_r < 0):
        return _convex_hull_mesh(pts)
    return verts, faces_r


def compute_tight_hull_payload(
    pts: np.ndarray,
    rng: np.random.Generator,
    max_pts: int,
    percentile: float,
) -> dict[str, Any] | None:
    if pts.shape[0] < 4:
        return None
    sub = _subsample_points(pts, max_pts, rng)
    mesh = _alpha_shape_boundary(sub, percentile)
    if mesh is None:
        return None
    v, f = mesh
    if f.size < 3:
        return None
    return _pack_mesh(v, f)


def _npz_meta_str(z: np.lib.npyio.NpzFile, key: str) -> str:
    a = np.asarray(z[key])
    if a.shape == ():
        return str(a.item())
    return str(a.ravel()[0])


def _assert_global_npz_matches_bundle(
    z: np.lib.npyio.NpzFile,
    *,
    window_size: int,
    stride: int,
    grain_mode: GrainMode,
    d_rows: int,
) -> None:
    fp_expect = grain_fingerprint(window_size, stride, grain_mode)
    fp_got = _npz_meta_str(z, "fingerprint")
    if fp_got != fp_expect:
        raise ValueError(
            f"Global PCA npz fingerprint {fp_got!r} != expected {fp_expect!r} "
            f"(window={window_size} stride={stride} mode={grain_mode}). "
            "Run scripts/export_global_grain_embedding.py with matching params."
        )
    d = int(z["d"])
    if d != d_rows:
        raise ValueError(f"Global npz dimension D={d} but grains have D={d_rows}")
    if "window_size" in z.files and int(z["window_size"]) != window_size:
        raise ValueError("Global npz window_size does not match bundle.")
    if "stride" in z.files and int(z["stride"]) != stride:
        raise ValueError("Global npz stride does not match bundle.")


def _make_traj_blocks(
    cb_xyz: np.ndarray,
    tr_xyz: np.ndarray,
    am_xyz: np.ndarray,
    *,
    n_sub: int,
    m: int,
    cb_rgb: np.ndarray,
    tr_rgb: np.ndarray,
    rng_h: np.random.Generator,
    tight_hulls: bool,
    hull_cb_max_pts: int,
    hull_traj_max_pts: int,
    hull_alpha_percentile: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    codebook_block: dict[str, Any] = {
        "count": n_sub,
        "pos_b64": _b64_f32(cb_xyz.reshape(-1)),
        "rgb_b64": _b64_f32(cb_rgb.reshape(-1)),
    }
    target_block: dict[str, Any] = {
        "count": m,
        "pos_b64": _b64_f32(tr_xyz.reshape(-1)),
        "rgb_b64": _b64_f32(tr_rgb.reshape(-1)),
    }
    argmax_block: dict[str, Any] = {
        "count": m,
        "pos_b64": _b64_f32(am_xyz.reshape(-1)),
    }
    if tight_hulls:
        h_cb = compute_tight_hull_payload(cb_xyz, rng_h, hull_cb_max_pts, hull_alpha_percentile)
        if h_cb:
            codebook_block["tight_hull"] = h_cb
        h_tr = compute_tight_hull_payload(tr_xyz, rng_h, hull_traj_max_pts, hull_alpha_percentile)
        if h_tr:
            target_block["tight_hull"] = h_tr
        h_am = compute_tight_hull_payload(am_xyz, rng_h, hull_traj_max_pts, hull_alpha_percentile)
        if h_am:
            argmax_block["tight_hull"] = h_am
    return codebook_block, target_block, argmax_block


def _time_colors(n: int) -> np.ndarray:
    c = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        u = i / max(1, n - 1)
        c[i, 0] = 0.12 + 0.75 * u
        c[i, 1] = 0.05 + 0.85 * (0.5 + 0.5 * np.sin(u * np.pi))
        c[i, 2] = 0.35 + 0.55 * (1.0 - u)
    return c


def bundle_id(source_folder: str, target_folder: str, target_stem: str) -> str:
    """Stable filename stem for JSON (no path separators)."""
    def clean(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]", "_", s)

    return f"{clean(source_folder)}__{clean(target_folder)}__{clean(target_stem)}"


def parse_target_npy(enc_root: str, target_npy: str) -> tuple[str, str]:
    """Return (target_folder, stem_without_suffix)."""
    ap = os.path.abspath(target_npy)
    er = os.path.abspath(enc_root)
    dn = os.path.dirname(ap)
    stem = os.path.splitext(os.path.basename(ap))[0]
    try:
        rel = os.path.relpath(dn, er)
    except ValueError:
        rel = os.path.basename(dn)
    parts = rel.split(os.sep)
    target_folder = parts[0] if parts and parts[0] not in (".", "") else os.path.basename(dn)
    return target_folder, stem


def scan_enc_tree(enc_root: str) -> dict[str, Any]:
    """List category folders and .npy stems under enc_root."""
    if not os.path.isdir(enc_root):
        raise FileNotFoundError(enc_root)
    folders = sorted(
        d for d in os.listdir(enc_root) if os.path.isdir(os.path.join(enc_root, d))
    )
    targets: list[dict[str, str]] = []
    for folder in folders:
        dpath = os.path.join(enc_root, folder)
        for fname in sorted(f for f in os.listdir(dpath) if f.endswith(".npy")):
            stem = os.path.splitext(fname)[0]
            targets.append({"folder": folder, "stem": stem, "label": f"{folder}/{stem}"})
    return {"enc_root": enc_root, "folders": folders, "targets": targets}


def discover_bundles(bundles_dir: str) -> list[str]:
    if not os.path.isdir(bundles_dir):
        return []
    out: list[str] = []
    for fname in os.listdir(bundles_dir):
        if fname.endswith(".json") and fname != "manifest.json":
            out.append(os.path.splitext(fname)[0])
    return sorted(out)


def write_manifest(
    bundles_dir: str,
    enc_root: str,
    bundles_available: list[str] | None = None,
    *,
    global_corpus_cloud: str | None = None,
) -> str:
    os.makedirs(bundles_dir, exist_ok=True)
    tree = scan_enc_tree(enc_root)
    if bundles_available is None:
        bundles_available = discover_bundles(bundles_dir)
    manifest: dict[str, Any] = {
        "version": 1,
        **tree,
        "bundles_available": sorted(set(bundles_available)),
    }
    if global_corpus_cloud:
        manifest["global_corpus_cloud"] = global_corpus_cloud
    path = os.path.join(bundles_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=0)
    return path


def pack_bundle_meta(
    C_full: np.ndarray,
    C_sub: np.ndarray,
    Tg: np.ndarray,
    *,
    source_folder: str,
    tgt_folder: str,
    tgt_stem: str,
    window_size: int,
    stride: int,
    grain_mode: GrainMode,
    seed: int,
    tight_hulls: bool = True,
    hull_cb_max_pts: int = 3500,
    hull_traj_max_pts: int = 4000,
    hull_alpha_percentile: float = 28.0,
    global_embed_npz: str | None = None,
    global_umap_joblib: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Joint PCA(3) + trajectories; optional global PCA / UMAP 3D (same L2-normalized grains)."""
    n_full = int(C_full.shape[0])
    n_sub = int(C_sub.shape[0])
    m = int(Tg.shape[0])
    d_rows = int(C_sub.shape[1])
    Z_fit = np.vstack([C_sub.astype(np.float64), Tg.astype(np.float64)])
    pca_joint = PCA(n_components=3, random_state=seed)
    pca_joint.fit(Z_fit)
    cb_xyz = pca_joint.transform(C_sub.astype(np.float64)).astype(np.float32)
    tr_xyz = pca_joint.transform(Tg.astype(np.float64)).astype(np.float32)
    sims = Tg @ C_full.T
    arg_idx = np.argmax(sims, axis=1).astype(np.int64)
    am_xyz = pca_joint.transform(C_full[arg_idx].astype(np.float64)).astype(np.float32)

    cb_xyz = _normalize_xyz(cb_xyz)
    tr_xyz = _normalize_xyz(tr_xyz)
    am_xyz = _normalize_xyz(am_xyz)

    cb_rgb = np.full((n_sub, 3), 0.55, dtype=np.float32)
    cb_rgb[:, 0] = 0.62
    cb_rgb[:, 1] = 0.64
    cb_rgb[:, 2] = 0.68
    tr_rgb = _time_colors(m)

    bid = bundle_id(source_folder, tgt_folder, tgt_stem)
    rng_h = np.random.default_rng(seed + 2027)

    codebook_block, target_block, argmax_block = _make_traj_blocks(
        cb_xyz,
        tr_xyz,
        am_xyz,
        n_sub=n_sub,
        m=m,
        cb_rgb=cb_rgb,
        tr_rgb=tr_rgb,
        rng_h=rng_h,
        tight_hulls=tight_hulls,
        hull_cb_max_pts=hull_cb_max_pts,
        hull_traj_max_pts=hull_traj_max_pts,
        hull_alpha_percentile=hull_alpha_percentile,
    )

    spaces: dict[str, Any] = {
        "joint_pca": {
            "codebook": codebook_block,
            "target_traj": target_block,
            "argmax_traj": argmax_block,
        }
    }

    g_npz = (global_embed_npz or "").strip()
    u_job = (global_umap_joblib or "").strip()
    if g_npz and os.path.isfile(g_npz):
        with np.load(g_npz, allow_pickle=False) as z:
            _assert_global_npz_matches_bundle(
                z,
                window_size=window_size,
                stride=stride,
                grain_mode=grain_mode,
                d_rows=d_rows,
            )
            mean = np.asarray(z["mean"], dtype=np.float64).ravel()
            comp = np.asarray(z["components"], dtype=np.float64)
            if comp.shape != (3, d_rows):
                raise ValueError(f"PCA components shape {comp.shape} expected (3, {d_rows})")

            if "corpus_norm_pca_center" in z.files:
                pca_ctr = np.asarray(z["corpus_norm_pca_center"], dtype=np.float64).ravel()
                pca_span = float(np.asarray(z["corpus_norm_pca_span"]))
            else:
                pca_ctr, pca_span = None, None  # type: ignore[assignment]

            def _global_pca_xyz(X: np.ndarray) -> np.ndarray:
                t = (X.astype(np.float64) - mean) @ comp.T
                if pca_ctr is not None:
                    return _apply_normalize_xyz(t, pca_ctr, pca_span)
                return _normalize_xyz(t.astype(np.float32))

            cb_g = _global_pca_xyz(C_sub)
            tr_g = _global_pca_xyz(Tg)
            am_g = _global_pca_xyz(C_full[arg_idx])
            c2, t2, a2 = _make_traj_blocks(
                cb_g,
                tr_g,
                am_g,
                n_sub=n_sub,
                m=m,
                cb_rgb=cb_rgb,
                tr_rgb=tr_rgb,
                rng_h=rng_h,
                tight_hulls=False,
                hull_cb_max_pts=hull_cb_max_pts,
                hull_traj_max_pts=hull_traj_max_pts,
                hull_alpha_percentile=hull_alpha_percentile,
            )
            spaces["global_pca"] = {"codebook": c2, "target_traj": t2, "argmax_traj": a2}

    if u_job and os.path.isfile(u_job):
        if joblib is None:
            raise ImportError("joblib required for --global-umap-model")
        if "global_pca" not in spaces:
            raise ValueError("global_umap_joblib requires a matching --global-embed-npz (fingerprint + D).")
        umap_ctr, umap_span = None, None  # type: ignore[assignment]
        if g_npz and os.path.isfile(g_npz):
            with np.load(g_npz, allow_pickle=False) as z2:
                if "corpus_norm_umap_center" in z2.files:
                    umap_ctr = np.asarray(z2["corpus_norm_umap_center"], dtype=np.float64).ravel()
                    umap_span = float(np.asarray(z2["corpus_norm_umap_span"]))
        reducer = joblib.load(u_job)

        def _norm_u(arr: np.ndarray) -> np.ndarray:
            if umap_ctr is not None:
                return _apply_normalize_xyz(arr, umap_ctr, umap_span)
            return _normalize_xyz(arr)

        cb_u = _norm_u(np.asarray(reducer.transform(C_sub.astype(np.float32)), dtype=np.float32))
        tr_u = _norm_u(np.asarray(reducer.transform(Tg.astype(np.float32)), dtype=np.float32))
        am_u = _norm_u(np.asarray(reducer.transform(C_full[arg_idx].astype(np.float32)), dtype=np.float32))
        c3, t3, a3 = _make_traj_blocks(
            cb_u,
            tr_u,
            am_u,
            n_sub=n_sub,
            m=m,
            cb_rgb=cb_rgb,
            tr_rgb=tr_rgb,
            rng_h=rng_h,
            tight_hulls=False,
            hull_cb_max_pts=hull_cb_max_pts,
            hull_traj_max_pts=hull_traj_max_pts,
            hull_alpha_percentile=hull_alpha_percentile,
        )
        spaces["global_umap"] = {"codebook": c3, "target_traj": t3, "argmax_traj": a3}

    notes_paths = (
        "Argmax-on-grains latent re-synthesis: glass tubes; "
        "alpha-shape tight_hull only for joint PCA (not global spaces)."
    )
    notes_emb = (
        f"Joint PCA(3) on stacked [codebook subsample n={n_sub}, full n={n_full}] and target grains (M={m}); "
        f"window={window_size} stride={stride} mode={grain_mode}."
    )
    if "global_pca" in spaces:
        notes_emb += " Global PCA(3) from pooled grains (export_global_grain_embedding.py)."
    if "global_umap" in spaces:
        notes_emb += " Global UMAP(3) transform of the same grain vectors."

    spaces_meta: dict[str, str] = {}
    if "joint_pca" in spaces:
        spaces_meta["joint_pca"] = (
            "Pairwise PCA(3) on stacked [codebook subsample, target grains]; "
            "alpha-shape tight_hull only here when enabled at export."
        )
    if "global_pca" in spaces:
        spaces_meta["global_pca"] = (
            "Corpus-wide PCA(3): background cloud (per-folder color + marker) when "
            "`manifest.global_corpus_cloud` is present; trajectories use the same normalization as the export."
        )
    if "global_umap" in spaces:
        spaces_meta["global_umap"] = (
            "Corpus UMAP(3): same as global PCA — full grain cloud in the viewer plus target/argmax tubes."
        )

    meta: dict[str, Any] = {
        "bundle_id": bid,
        "source_folder": source_folder,
        "target_folder": tgt_folder,
        "target_stem": tgt_stem,
        "target_label": f"{tgt_folder}/{tgt_stem}",
        "title": f"Latent granular 3D — {source_folder} → {tgt_folder}/{tgt_stem}",
        "notes": {
            "embedding": notes_emb,
            "paths": notes_paths,
            "decode": "Audio not rendered; this is EnCodec latent space only.",
        },
        "spaces_meta": spaces_meta,
        "spaces": spaces,
        "spaces_available": list(spaces.keys()),
        "codebook": codebook_block,
        "target_traj": target_block,
        "argmax_traj": argmax_block,
    }
    return meta, bid


def compute_bundle(
    enc_root: str,
    source_folder: str,
    target_npy: str,
    source_glob: str,
    max_source_files: int | None,
    window_size: int,
    stride: int,
    grain_mode: GrainMode,
    assume_row_is_time: bool,
    max_codebook_points: int,
    seed: int,
    *,
    tight_hulls: bool = True,
    hull_cb_max_pts: int = 3500,
    hull_traj_max_pts: int = 4000,
    hull_alpha_percentile: float = 28.0,
    global_embed_npz: str | None = None,
    global_umap_joblib: str | None = None,
) -> tuple[dict[str, Any], str]:
    assume_row = assume_row_is_time
    rng = np.random.default_rng(seed)

    codebook, _ = collect_source_codebook(
        enc_root,
        source_folder,
        glob_pattern=source_glob,
        max_files=max_source_files,
        window_size=window_size,
        stride=stride,
        grain_mode=grain_mode,
        assume_row_is_time=assume_row,
    )
    tx = load_embedding(target_npy, assume_row_is_time=assume_row)
    if window_size == 1 and stride == 1:
        target = tx
    else:
        target = make_grains(tx, window_size, stride, grain_mode)

    C_full = l2_normalize_rows(codebook.astype(np.float32))
    Tg = l2_normalize_rows(target.astype(np.float32))
    n_full = C_full.shape[0]

    if n_full > max_codebook_points:
        pick = rng.choice(n_full, size=max_codebook_points, replace=False)
        C_sub = C_full[pick]
    else:
        C_sub = C_full

    tgt_folder, tgt_stem = parse_target_npy(enc_root, target_npy)
    return pack_bundle_meta(
        C_full,
        C_sub,
        Tg,
        source_folder=source_folder,
        tgt_folder=tgt_folder,
        tgt_stem=tgt_stem,
        window_size=window_size,
        stride=stride,
        grain_mode=grain_mode,
        seed=seed,
        tight_hulls=tight_hulls,
        hull_cb_max_pts=hull_cb_max_pts,
        hull_traj_max_pts=hull_traj_max_pts,
        hull_alpha_percentile=hull_alpha_percentile,
        global_embed_npz=global_embed_npz,
        global_umap_joblib=global_umap_joblib,
    )


def build_all_bundles(
    enc_root: str,
    bundles_dir: str,
    *,
    source_glob: str,
    max_source_files: int | None,
    window_size: int,
    stride: int,
    grain_mode: GrainMode,
    assume_row_is_time: bool,
    max_codebook_points: int,
    seed: int,
    skip_existing: bool,
    progress_every: int,
    tight_hulls: bool = True,
    hull_cb_max_pts: int = 3500,
    hull_traj_max_pts: int = 4000,
    hull_alpha_percentile: float = 28.0,
    global_embed_npz: str | None = None,
    global_umap_joblib: str | None = None,
) -> tuple[int, int, int]:
    """For each folder as codebook source and each .npy as target, write one bundle JSON.
    Returns (written, skipped_existing, failed)."""
    tree = scan_enc_tree(enc_root)
    folders = tree["folders"]
    targets = tree["targets"]
    rng_src = np.random.default_rng(seed)
    written = 0
    skipped = 0
    failed = 0
    total_pairs = len(folders) * len(targets)
    pair_i = 0

    os.makedirs(bundles_dir, exist_ok=True)

    for source_folder in folders:
        try:
            codebook, _ = collect_source_codebook(
                enc_root,
                source_folder,
                glob_pattern=source_glob,
                max_files=max_source_files,
                window_size=window_size,
                stride=stride,
                grain_mode=grain_mode,
                assume_row_is_time=assume_row_is_time,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[build-all] skip source {source_folder!r}: {e}")
            continue

        C_full = l2_normalize_rows(codebook.astype(np.float32))
        n_full = C_full.shape[0]
        if n_full > max_codebook_points:
            pick = rng_src.choice(n_full, size=max_codebook_points, replace=False)
            C_sub = C_full[pick]
        else:
            C_sub = C_full

        for t in targets:
            pair_i += 1
            tgt_folder = t["folder"]
            tgt_stem = t["stem"]
            bid = bundle_id(source_folder, tgt_folder, tgt_stem)
            outp = os.path.join(bundles_dir, bid + ".json")
            if skip_existing and os.path.isfile(outp):
                skipped += 1
                if progress_every and pair_i % progress_every == 0:
                    print(f"[build-all] {pair_i}/{total_pairs} (written={written} skip={skipped} fail={failed})")
                continue

            target_path = os.path.join(enc_root, tgt_folder, tgt_stem + ".npy")
            try:
                tx = load_embedding(target_path, assume_row_is_time=assume_row_is_time)
                if window_size == 1 and stride == 1:
                    raw_t = tx
                else:
                    raw_t = make_grains(tx, window_size, stride, grain_mode)
                Tg = l2_normalize_rows(raw_t.astype(np.float32))
            except Exception as e:  # noqa: BLE001
                print(f"[build-all] fail {bid}: {e}")
                failed += 1
                continue

            try:
                meta, _ = pack_bundle_meta(
                    C_full,
                    C_sub,
                    Tg,
                    source_folder=source_folder,
                    tgt_folder=tgt_folder,
                    tgt_stem=tgt_stem,
                    window_size=window_size,
                    stride=stride,
                    grain_mode=grain_mode,
                    seed=seed,
                    tight_hulls=tight_hulls,
                    hull_cb_max_pts=hull_cb_max_pts,
                    hull_traj_max_pts=hull_traj_max_pts,
                    hull_alpha_percentile=hull_alpha_percentile,
                    global_embed_npz=global_embed_npz,
                    global_umap_joblib=global_umap_joblib,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[build-all] fail {bid} (pack): {e}")
                failed += 1
                continue

            with open(outp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=0)
            written += 1
            if progress_every and pair_i % progress_every == 0:
                print(f"[build-all] {pair_i}/{total_pairs} (written={written} skip={skipped} fail={failed})")

    print(
        f"[build-all] done: wrote={written}, skipped_existing={skipped}, failed={failed}, "
        f"pairs_touched={pair_i}/{total_pairs}"
    )
    return written, skipped, failed


VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en" data-viewer-revision="35" data-corpus-marker-scale="0.026">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="Cache-Control" content="no-store, max-age=0" />
  <title id="docTitle">Latent granular 3D</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --fa-bg: #0a0a09;
      --fa-panel: #0f0f0e;
      --fa-raised: #151514;
      --fa-border: #2c2b28;
      --fa-line: #3a3834;
      --fa-text: #eceae5;
      --fa-muted: #8f8c85;
      --fa-dim: #5a5853;
      --fa-accent: #d63031;
      --fa-field: #060605;
    }
    html, body {
      margin: 0; height: 100%; overflow: hidden;
      background: var(--fa-bg);
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
      font-size: 13px;
      color: var(--fa-text);
    }
    #bar {
      position: fixed; top: 0; left: 0; z-index: 10;
      width: min(400px, 100vw);
      max-height: 100vh;
      overflow-y: auto;
      overflow-x: hidden;
      box-sizing: border-box;
      scrollbar-width: thin;
      scrollbar-color: var(--fa-line) var(--fa-panel);
      line-height: 1.45;
      color: var(--fa-text);
      background: var(--fa-panel);
      border-right: 1px solid var(--fa-border);
      border-bottom: 1px solid var(--fa-border);
      padding: 0;
      box-shadow: 6px 0 40px rgba(0, 0, 0, 0.4);
    }
    #bar.fa-bar-light {
      --fa-panel: #f2f0eb;
      --fa-raised: #e8e6e0;
      --fa-border: #c4c0b6;
      --fa-line: #a39e94;
      --fa-text: #141413;
      --fa-muted: #5e5c58;
      --fa-dim: #8a8680;
      --fa-field: #ffffff;
      box-shadow: 6px 0 28px rgba(0, 0, 0, 0.06);
    }
    #bar::-webkit-scrollbar { width: 7px; }
    #bar::-webkit-scrollbar-track { background: var(--fa-panel); }
    #bar::-webkit-scrollbar-thumb { background: var(--fa-line); }
    #bar .bar-body { padding: 14px 16px 20px; }
    #bar h1 {
      margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.02em;
      font-family: 'IBM Plex Sans', sans-serif;
    }
    #bar .row { margin-top: 12px; }
    #bar label.lbl, #bar .section-hdr {
      display: block;
      font-family: 'IBM Plex Mono', ui-monospace, monospace;
      font-size: 10px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--fa-muted);
      margin-bottom: 6px;
    }
    #bar select {
      width: 100%; box-sizing: border-box;
      padding: 8px 10px;
      border: 1px solid var(--fa-border);
      border-radius: 0;
      background: var(--fa-field);
      color: var(--fa-text);
      font: inherit;
      cursor: pointer;
    }
    #bar select:focus, #bar input:focus, #bar button:focus { outline: 1px solid var(--fa-accent); outline-offset: 1px; }
    #bar select:hover { border-color: var(--fa-line); }
    #bar select.multi { min-height: 7rem; padding: 6px 8px; line-height: 1.35; }
    #bar p.meta {
      margin: 8px 0 0; color: var(--fa-muted);
      font-size: 11px; line-height: 1.5;
      font-family: 'IBM Plex Sans', sans-serif;
    }
    #bar label.cb {
      display: flex; align-items: flex-start; gap: 10px;
      margin-top: 10px; cursor: pointer; font-size: 12px;
      line-height: 1.35; color: var(--fa-text);
    }
    #bar label.fa-row-first { margin-top: 0; }
    #bar label.cb input { margin-top: 2px; accent-color: var(--fa-accent); }
    #bar .corpus-folder-checks {
      max-height: 11rem; overflow-y: auto; margin-top: 8px;
      padding: 2px 0; border: 1px solid var(--fa-border);
      background: var(--fa-field);
    }
    #bar .corpus-folder-checks label.cb { margin-top: 6px; margin-left: 10px; margin-right: 10px; }
    #bar .corpus-folder-checks label.cb:first-child { margin-top: 10px; }
    #bar .anim-row {
      margin-top: 12px; display: grid; grid-template-columns: 1fr auto;
      gap: 10px; align-items: center;
    }
    #bar .anim-row label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--fa-muted);
    }
    #bar input[type="range"] {
      width: 100%; height: 4px; margin: 4px 0 0;
      accent-color: var(--fa-accent);
      background: transparent;
    }
    #bar .anim-val {
      font-family: 'IBM Plex Mono', monospace;
      font-variant-numeric: tabular-nums;
      color: var(--fa-text); font-size: 11px;
      min-width: 40px; text-align: right;
    }
    #status {
      margin-top: 16px; padding: 10px 12px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; line-height: 1.45;
      text-transform: uppercase; letter-spacing: 0.08em;
      background: var(--fa-field);
      border: 1px solid var(--fa-border);
      color: var(--fa-muted);
    }
    #status.err { border-color: var(--fa-accent); color: var(--fa-accent); background: var(--fa-panel); }
    #status.ok { border-color: var(--fa-dim); color: var(--fa-muted); }
    .bar-top {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--fa-border);
      background: var(--fa-raised);
    }
    .bar-top h1 { flex: 1; min-width: 0; }
    .bar-collapse-btn {
      flex-shrink: 0;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em;
      cursor: pointer; padding: 8px 12px;
      border: 1px solid var(--fa-border); border-radius: 0;
      background: var(--fa-field); color: var(--fa-text);
    }
    .bar-collapse-btn:hover { border-color: var(--fa-accent); color: var(--fa-accent); }
    #bar.bar-panel-collapsed .bar-body { display: none; }
    #bar.bar-panel-collapsed { max-height: none; }
    .fa-details {
      margin-top: 14px;
      border: 1px solid var(--fa-border);
      background: var(--fa-field);
    }
    .fa-details summary {
      list-style: none;
      cursor: pointer;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; font-weight: 500;
      text-transform: uppercase; letter-spacing: 0.16em;
      color: var(--fa-text);
      padding: 11px 14px;
      background: var(--fa-raised);
      border-bottom: 1px solid var(--fa-border);
      user-select: none;
    }
    .fa-details summary::-webkit-details-marker { display: none; }
    .fa-details summary::before {
      content: '+';
      display: inline-block; width: 1.2em;
      color: var(--fa-accent); font-weight: 600;
    }
    .fa-details[open] summary::before { content: '−'; }
    .fa-details .fa-details-inner { padding: 6px 14px 14px; }
    .fa-details-inner > .section-hdr:first-child { margin-top: 0; }
    .umap-legend-window {
      position: fixed; right: 0; bottom: 0; z-index: 12;
      max-width: min(280px, 42vw);
      max-height: min(50vh, 380px);
      overflow-y: auto; overflow-x: hidden;
      box-sizing: border-box;
      padding: 14px 16px;
      border: 1px solid var(--fa-border);
      border-right: none; border-bottom: none;
      background: var(--fa-panel);
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 12px; line-height: 1.4;
      color: var(--fa-text);
      box-shadow: -6px -6px 40px rgba(0, 0, 0, 0.35);
      scrollbar-width: thin;
      scrollbar-color: var(--fa-line) var(--fa-panel);
    }
    .umap-legend-window::-webkit-scrollbar { width: 6px; }
    .umap-legend-window::-webkit-scrollbar-track { background: var(--fa-panel); }
    .umap-legend-window::-webkit-scrollbar-thumb { background: var(--fa-line); }
    .umap-legend-window.umap-legend-light {
      background: #f2f0eb;
      border-color: #c4c0b6;
      color: #141413;
      box-shadow: -6px -6px 24px rgba(0, 0, 0, 0.08);
      scrollbar-color: #a39e94 #f2f0eb;
    }
    .umap-legend-window.umap-legend-light::-webkit-scrollbar-thumb { background: #a39e94; }
    .umap-legend-window.umap-legend-light .umap-leg-row { color: #3d3c3a; }
    .umap-legend-window.umap-legend-light .umap-legend-win-hdr { color: #1a1918; }
    .umap-legend-window.umap-legend-light .umap-leg-swatch { border-color: #8a8680; }
    .umap-legend-win-hdr {
      display: block;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; font-weight: 500;
      text-transform: uppercase; letter-spacing: 0.16em;
      color: var(--fa-muted);
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--fa-border);
    }
    .umap-leg-row {
      display: flex; align-items: center; gap: 10px;
      margin-top: 8px; font-size: 11px; color: var(--fa-muted);
      font-family: 'IBM Plex Sans', sans-serif;
    }
    .umap-leg-swatch {
      width: 11px; height: 11px;
      border-radius: 0;
      flex-shrink: 0;
      border: 1px solid var(--fa-line);
    }
  </style>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
  }
  </script>
</head>
<body>
  <div id="bar">
    <div class="bar-top">
      <h1 id="hdrTitle">Latent granular 3D</h1>
      <button type="button" id="barCollapse" class="bar-collapse-btn" aria-expanded="true" aria-controls="barBody">Collapse</button>
    </div>
    <div id="barBody" class="bar-body">
    <label class="cb fa-row-first"><input type="checkbox" id="whiteBgCb" aria-label="Use white scene background" /> White background (canvas + page)</label>
    <div class="row">
      <span class="lbl">Codebook source (folder)</span>
      <select id="selSource" aria-label="Codebook source folder"></select>
    </div>
    <div class="row">
      <span class="lbl">Target category</span>
      <select id="selTargetFolder" aria-label="Target folder"></select>
    </div>
    <div class="row">
      <span class="lbl">Target sound</span>
      <select id="selTargetStem" aria-label="Target file stem"></select>
    </div>
    <div class="row" id="embedSpaceRow" style="display:none">
      <span class="lbl">Embedding space</span>
      <select id="embedSpace" aria-label="3D coordinates: joint vs global PCA/UMAP">
        <option value="joint_pca">Joint PCA (pair)</option>
      </select>
    </div>
    <div class="row" id="corpusFoldersRow" style="display:none">
      <span class="lbl">Corpus folders (global PCA/UMAP)</span>
      <div id="corpusFolderFilterMount" class="corpus-folder-checks" aria-label="Enc categories shown in corpus background"></div>
      <p class="meta" style="margin-top:4px">Check or uncheck categories to mix multiple folders in the corpus background.</p>
    </div>
    <div id="corpusVizRow" style="display:none">
      <p class="meta section-hdr" style="margin-top:8px">Corpus cloud (global PCA/UMAP)</p>
      <div class="anim-row">
        <label for="corpusOpacity">Point opacity</label>
        <span id="corpusOpacityVal" class="anim-val">1.00</span>
      </div>
      <input type="range" id="corpusOpacity" min="0.06" max="1" step="0.02" value="1" aria-label="Corpus marker opacity" />
      <div class="anim-row">
        <label for="corpusPointSize">Point size</label>
        <span id="corpusPointSizeVal" class="anim-val">0.026</span>
      </div>
      <input type="range" id="corpusPointSize" min="0.002" max="0.05" step="0.0005" value="0.026" aria-label="Corpus marker world scale" />
      <div class="anim-row">
        <label for="corpusSpread">Spread (separate points)</label>
        <span id="corpusSpreadVal" class="anim-val">10.00</span>
      </div>
      <input type="range" id="corpusSpread" min="0.5" max="10" step="0.05" value="10" aria-label="Scale corpus positions from origin" />
    </div>
    <label class="cb"><input type="checkbox" id="showTr" checked /> Target trajectory (time colour)</label>
    <label class="cb"><input type="checkbox" id="showAm" checked /> Argmax codebook walk</label>
    <details class="fa-details" id="tubePropsDetails" open>
      <summary>Tubes, lighting and motion</summary>
      <div class="fa-details-inner">
    <p class="meta section-hdr">Trajectory look</p>
    <div class="anim-row">
      <label for="trajRadTarget">Target tube radius</label>
      <span id="trajRadTargetVal" class="anim-val">0.002</span>
    </div>
    <input type="range" id="trajRadTarget" min="0.002" max="0.024" step="0.0005" value="0.002" aria-label="Radius of target path tube" />
    <div class="anim-row">
      <label for="trajRadArgmax">Argmax tube radius</label>
      <span id="trajRadArgmaxVal" class="anim-val">0.022</span>
    </div>
    <input type="range" id="trajRadArgmax" min="0.002" max="0.022" step="0.0005" value="0.022" aria-label="Radius of argmax path tube" />
    <div class="anim-row">
      <label for="trajPulse">Path highlight sharpness</label>
      <span id="trajPulseVal" class="anim-val">52</span>
    </div>
    <input type="range" id="trajPulse" min="8" max="52" step="1" value="52" aria-label="Tighter highlight along tube" />
    <div class="anim-row">
      <label for="trajAlpha">Tube opacity</label>
      <span id="trajAlphaVal" class="anim-val">0.00</span>
    </div>
    <input type="range" id="trajAlpha" min="0" max="1.6" step="0.05" value="0" aria-label="Glass tube opacity multiplier" />
    <div class="anim-row">
      <label for="trajHaze">Flow haze</label>
      <span id="trajHazeVal" class="anim-val">1.00</span>
    </div>
    <input type="range" id="trajHaze" min="0" max="1" step="0.02" value="1" aria-label="Soft drifting haze along trajectory tubes" />
    <label class="cb"><input type="checkbox" id="trajProxUmap" checked /> Light corpus near paths (dims tube glow; rest of UMAP fades to near-off for contrast)</label>
    <div class="anim-row">
      <label for="trajProxRadius">Corpus glow radius (color)</label>
      <span id="trajProxRadiusVal" class="anim-val">0.18</span>
    </div>
    <input type="range" id="trajProxRadius" min="0.04" max="1.2" step="0.01" value="0.18" aria-label="Falloff distance for corpus color lighting near path heads" />
    <div class="anim-row">
      <label for="trajProxSizeRadius">Codebook size boost radius</label>
      <span id="trajProxSizeRadiusVal" class="anim-val">0.24</span>
    </div>
    <input type="range" id="trajProxSizeRadius" min="0.04" max="1.2" step="0.01" value="0.24" aria-label="Falloff for enlarging codebook corpus markers near argmax head" />
    <label class="cb"><input type="checkbox" id="animPath" checked /> Animate paths (loop)</label>
    <div class="anim-row">
      <label for="animPeriod">Loop length (seconds)</label>
      <span id="animPeriodVal" class="anim-val">180</span>
    </div>
    <input type="range" id="animPeriod" min="0.5" max="180" step="0.5" value="180" aria-label="Seconds per animation loop (0.5–180)" />
      </div>
    </details>
    <div id="status" class="ok">Loading manifest…</div>
    </div>
  </div>
  <div id="umapLegendWin" class="umap-legend-window" style="display:none" role="complementary" aria-label="UMAP legend (separate panel)">
    <span class="umap-legend-win-hdr">UMAP legend</span>
    <div id="umapLegendSwatches" class="umap-legend-swatches" aria-label="Folder colors in corpus"></div>
  </div>
  <script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
  import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';

  const MANIFEST_URL = 'bundles/manifest.json';

  const bundleCache = new Map();
  const bundleInflight = new Map();
  const corpusCloudCache = new Map();
  const corpusCloudInflight = new Map();
  let manifestRef = null;

  function refitUmapLegend(folders, folderStyles) {
    const win = document.getElementById('umapLegendWin');
    const mount = document.getElementById('umapLegendSwatches');
    if (!win || !mount) return;
    if (!Array.isArray(folders) || !Array.isArray(folderStyles) || !folders.length) {
      win.style.display = 'none';
      return;
    }
    mount.innerHTML = '';
    for (let i = 0; i < folders.length; i++) {
      const row = document.createElement('div');
      row.className = 'umap-leg-row';
      const sw = document.createElement('span');
      sw.className = 'umap-leg-swatch';
      const st = folderStyles[i] || { color: '#94a3b8' };
      sw.style.background = st.color || '#94a3b8';
      const lab = document.createElement('span');
      lab.textContent = folders[i];
      row.appendChild(sw);
      row.appendChild(lab);
      mount.appendChild(row);
    }
    win.style.display = 'block';
  }

  function hideUmapLegend() {
    const win = document.getElementById('umapLegendWin');
    if (win) win.style.display = 'none';
  }

  function cleanBundlePart(s) {
    return String(s).replace(/[^a-zA-Z0-9._-]/g, '_');
  }

  function decodeF32(b64, n) {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Float32Array(raw.buffer, raw.byteOffset, n);
  }

  function decodeU8(b64, n) {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    if (raw.length < n) throw new Error('decodeU8: short buffer');
    return raw.subarray(0, n);
  }

  function parseHexRgb(hex) {
    let h = String(hex || '#94a3b8').trim();
    if (h.startsWith('#')) h = h.slice(1);
    if (h.length === 3) {
      h = h.split('').map((c) => c + c).join('');
    }
    const v = parseInt(h, 16);
    if (!Number.isFinite(v) || h.length !== 6) return { r: 0.58, g: 0.64, b: 0.72 };
    return { r: ((v >> 16) & 255) / 255, g: ((v >> 8) & 255) / 255, b: (v & 255) / 255 };
  }

  async function fetchCorpusCloudJson(url) {
    if (corpusCloudCache.has(url)) return corpusCloudCache.get(url);
    if (corpusCloudInflight.has(url)) return corpusCloudInflight.get(url);
    const p = fetch(url, { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((j) => {
        corpusCloudCache.set(url, j);
        corpusCloudInflight.delete(url);
        return j;
      })
      .catch((e) => {
        corpusCloudInflight.delete(url);
        throw e;
      });
    corpusCloudInflight.set(url, p);
    return p;
  }

  function corpusMarkerGeometry(which) {
    const k = ((which % 7) + 7) % 7;
    if (k === 0) return new THREE.BoxGeometry(1, 1, 1);
    if (k === 1) return new THREE.OctahedronGeometry(0.95, 0);
    if (k === 2) return new THREE.TetrahedronGeometry(1.02, 0);
    if (k === 3) return new THREE.IcosahedronGeometry(0.72, 0);
    if (k === 4) return new THREE.SphereGeometry(0.55, 10, 8);
    if (k === 5) return new THREE.CylinderGeometry(0.45, 0.45, 0.95, 10);
    return new THREE.ConeGeometry(0.55, 1.05, 10);
  }

  function wireCorpusFolderFilterMount() {
    const mount = document.getElementById('corpusFolderFilterMount');
    if (!mount || mount.dataset.filterDelegate === '1') return;
    mount.dataset.filterDelegate = '1';
    mount.addEventListener('change', (e) => {
      const t = e.target;
      if (!(t instanceof HTMLInputElement) || t.type !== 'checkbox') return;
      if (t.id === 'corpusFolderFilterAllCb') {
        const want = t.checked;
        t.indeterminate = false;
        mount.querySelectorAll('.corpus-folder-filter-cb').forEach((cb) => {
          cb.checked = want;
        });
      }
      if (t.id === 'corpusFolderFilterAllCb' || t.classList.contains('corpus-folder-filter-cb')) {
        onCorpusFolderFilterChange();
      }
    });
  }

  function syncCorpusFolderAllCheckbox() {
    const allEl = document.getElementById('corpusFolderFilterAllCb');
    const boxes = document.querySelectorAll('.corpus-folder-filter-cb');
    if (!allEl || !boxes.length) return;
    let n = 0;
    for (let i = 0; i < boxes.length; i++) {
      if (boxes[i].checked) n++;
    }
    allEl.indeterminate = n > 0 && n < boxes.length;
    allEl.checked = n === boxes.length;
  }

  function onCorpusFolderFilterChange() {
    syncCorpusFolderAllCheckbox();
    const key = document.getElementById('embedSpace').value;
    if (key !== 'global_pca' && key !== 'global_umap') return;
    applyEmbeddingSpace(key).catch((e) => {
      setStatus(String(e && e.message ? e.message : e), true);
    });
  }

  function refitCorpusFolderOptions(folders) {
    if (!Array.isArray(folders) || folders.length === 0) return;
    const mount = document.getElementById('corpusFolderFilterMount');
    if (!mount) return;
    const sig = folders.join('\0');
    if (mount.dataset.folderSig === sig && document.getElementById('corpusFolderFilterAllCb')) return;
    mount.dataset.folderSig = sig;
    mount.innerHTML = '';
    const allLab = document.createElement('label');
    allLab.className = 'cb';
    allLab.style.fontWeight = '600';
    const allInp = document.createElement('input');
    allInp.type = 'checkbox';
    allInp.id = 'corpusFolderFilterAllCb';
    allInp.checked = true;
    allInp.setAttribute('aria-label', 'Select or clear all corpus folders');
    allLab.appendChild(allInp);
    allLab.appendChild(document.createTextNode(' All'));
    mount.appendChild(allLab);
    for (let i = 0; i < folders.length; i++) {
      const lab = document.createElement('label');
      lab.className = 'cb';
      const inp = document.createElement('input');
      inp.type = 'checkbox';
      inp.className = 'corpus-folder-filter-cb';
      inp.id = 'cfcb-' + i;
      inp.dataset.fid = String(i);
      inp.checked = true;
      lab.appendChild(inp);
      lab.appendChild(document.createTextNode(' ' + folders[i]));
      mount.appendChild(lab);
    }
    syncCorpusFolderAllCheckbox();
  }

  function getCorpusFolderFilterSet() {
    const boxes = document.querySelectorAll('.corpus-folder-filter-cb');
    if (!boxes.length) return null;
    const s = new Set();
    for (let i = 0; i < boxes.length; i++) {
      const cb = boxes[i];
      if (cb.checked) s.add(parseInt(cb.dataset.fid, 10));
    }
    if (s.size === 0) return s;
    if (s.size === boxes.length) return null;
    return s;
  }

  function getCorpusVizFromUI() {
    const opEl = document.getElementById('corpusOpacity');
    const psEl = document.getElementById('corpusPointSize');
    const spEl = document.getElementById('corpusSpread');
    const op = opEl ? parseFloat(opEl.value) : NaN;
    const ps = psEl ? parseFloat(psEl.value) : NaN;
    const sp = spEl ? parseFloat(spEl.value) : NaN;
    return {
      opacity: Number.isFinite(op) ? op : 1.0,
      pointScale: Number.isFinite(ps) ? ps : 0.026,
      spread: Number.isFinite(sp) ? sp : 10.0,
    };
  }

  function buildCorpusCloudGroup(spaceBlock, folderStyles, folderFilter, vizIn) {
    const viz = vizIn || getCorpusVizFromUI();
    const useFilter = folderFilter !== null && folderFilter !== undefined;
    const n = spaceBlock.count | 0;
    const cpos = decodeF32(spaceBlock.pos_b64, n * 3);
    const fidx = decodeU8(spaceBlock.folder_idx_b64, n);
    const group = new THREE.Group();
    group.renderOrder = 0;
    const dummy = new THREE.Object3D();
    const byFolder = new Map();
    const s = viz.spread;
    for (let i = 0; i < n; i++) {
      const f = fidx[i];
      if (useFilter && !folderFilter.has(f)) continue;
      if (!byFolder.has(f)) byFolder.set(f, []);
      byFolder.get(f).push(i);
    }
    const scale = viz.pointScale;
    for (const [f, idxs] of byFolder) {
      const st = folderStyles[f] || { color: '#94a3b8', marker: 0 };
      const mid = st.marker | 0;
      const geo = corpusMarkerGeometry(mid);
      const { r, g, b } = parseHexRgb(st.color);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        metalness: 0.22,
        roughness: 0.48,
        transparent: true,
        opacity: viz.opacity,
        depthWrite: false,
        envMapIntensity: 0.95,
      });
      const inst = new THREE.InstancedMesh(geo, mat, idxs.length);
      inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const glowPos = new Float32Array(idxs.length * 3);
      const icGlow = new THREE.InstancedBufferAttribute(new Float32Array(idxs.length * 3), 3);
      icGlow.setUsage(THREE.DynamicDrawUsage);
      for (let t = 0; t < idxs.length * 3; t++) icGlow.array[t] = 1;
      inst.instanceColor = icGlow;
      inst.userData.corpusGlowPos = glowPos;
      inst.userData.corpusFolderId = f;
      inst.userData.corpusBaseScale = scale;
      for (let j = 0; j < idxs.length; j++) {
        const i = idxs[j];
        const o = i * 3;
        const wx = cpos[o] * s;
        const wy = cpos[o + 1] * s;
        const wz = cpos[o + 2] * s;
        glowPos[j * 3] = wx;
        glowPos[j * 3 + 1] = wy;
        glowPos[j * 3 + 2] = wz;
        dummy.position.set(wx, wy, wz);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();
        inst.setMatrixAt(j, dummy.matrix);
      }
      inst.instanceMatrix.needsUpdate = true;
      inst.frustumCulled = true;
      inst.renderOrder = 0;
      group.add(inst);
    }
    return group;
  }

  function manifestFolderNameToIndex(name) {
    if (!manifestRef || !Array.isArray(manifestRef.folders) || !name) return -1;
    const i = manifestRef.folders.indexOf(String(name));
    return i >= 0 ? i : -1;
  }

  function corpusHighlightUsesLightBg() {
    const el = document.getElementById('whiteBgCb');
    return !!(el && el.checked);
  }

  function resetCorpusProximityColors() {
    if (!cloud) return;
    const dummy = new THREE.Object3D();
    cloud.traverse((obj) => {
      if (!obj.isInstancedMesh || !obj.instanceColor || !obj.userData.corpusGlowPos) return;
      const ar = obj.instanceColor.array;
      for (let i = 0; i < ar.length; i++) ar[i] = 1;
      obj.instanceColor.needsUpdate = true;
      const bs = obj.userData.corpusBaseScale;
      if (Number.isFinite(bs)) {
        const pos = obj.userData.corpusGlowPos;
        const nInst = obj.count;
        for (let j = 0; j < nInst; j++) {
          const o = j * 3;
          dummy.position.set(pos[o], pos[o + 1], pos[o + 2]);
          dummy.rotation.set(0, 0, 0);
          dummy.scale.setScalar(bs);
          dummy.updateMatrix();
          obj.setMatrixAt(j, dummy.matrix);
        }
        obj.instanceMatrix.needsUpdate = true;
      }
    });
  }

  function updateCorpusProximityGlow(headTx, headTy, headTz, headAx, headAy, headAz) {
    const proxEl = document.getElementById('trajProxUmap');
    const enabled = !!(proxEl && proxEl.checked);
    const key = document.getElementById('embedSpace').value;
    const isGlobalCorpus =
      (key === 'global_pca' || key === 'global_umap')
      && manifestRef
      && manifestRef.global_corpus_cloud;
    if (!enabled || !isGlobalCorpus || !cloud) {
      if (corpusProxWasOn) resetCorpusProximityColors();
      corpusProxWasOn = false;
      return;
    }
    corpusProxWasOn = true;
    const srcIdx = manifestFolderNameToIndex(document.getElementById('selSource').value);
    const tgtIdx = manifestFolderNameToIndex(document.getElementById('selTargetFolder').value);
    const radEl = document.getElementById('trajProxRadius');
    let radiusGlow = radEl ? parseFloat(radEl.value) : NaN;
    if (!Number.isFinite(radiusGlow) || radiusGlow < 1e-6) radiusGlow = 0.18;
    const invR2Glow = 1 / (radiusGlow * radiusGlow);
    const radSizeEl = document.getElementById('trajProxSizeRadius');
    let radiusSize = radSizeEl ? parseFloat(radSizeEl.value) : NaN;
    if (!Number.isFinite(radiusSize) || radiusSize < 1e-6) radiusSize = 0.24;
    const invR2Size = 1 / (radiusSize * radiusSize);
    const cap = 2.45;
    const lightBg = corpusHighlightUsesLightBg();
    const dimUnrelated = lightBg ? 0.05 : 0.11;
    const dimTrackedIdle = lightBg ? 0.14 : 0.15;
    const dummy = new THREE.Object3D();
    const uTime = performance.now() * 0.001;

    cloud.traverse((obj) => {
      if (!obj.isInstancedMesh || !obj.instanceColor || !obj.userData.corpusGlowPos) return;
      const folderId = obj.userData.corpusFolderId | 0;
      const pos = obj.userData.corpusGlowPos;
      const nInst = obj.count;
      const ar = obj.instanceColor.array;
      const inTgt = tgtIdx >= 0 && folderId === tgtIdx;
      const inSrc = srcIdx >= 0 && folderId === srcIdx;
      for (let j = 0; j < nInst; j++) {
        const o = j * 3;
        const px = pos[o];
        const py = pos[o + 1];
        const pz = pos[o + 2];
        if (!inTgt && !inSrc) {
          ar[o] = dimUnrelated;
          ar[o + 1] = dimUnrelated;
          ar[o + 2] = dimUnrelated;
          continue;
        }
        let fT = 0;
        let fA = 0;
        if (inTgt) {
          const dx = px - headTx;
          const dy = py - headTy;
          const dz = pz - headTz;
          fT = Math.exp(-(dx * dx + dy * dy + dz * dz) * invR2Glow);
        }
        if (inSrc) {
          const dx = px - headAx;
          const dy = py - headAy;
          const dz = pz - headAz;
          fA = Math.exp(-(dx * dx + dy * dy + dz * dz) * invR2Glow);
        }
        let rr; let gg; let bb;
        if (lightBg) {
          rr = dimTrackedIdle + fT * 0.14 + fA * 0.3;
          gg = dimTrackedIdle + fT * 0.22 + fA * 0.13;
          bb = dimTrackedIdle + fT * 0.28 + fA * 0.07;
          const m = Math.min(cap, 0.52);
          rr = Math.min(rr, m);
          gg = Math.min(gg, m);
          bb = Math.min(bb, m);
        } else {
          rr = dimTrackedIdle + fT * 0.5 + fA * 2.05;
          gg = dimTrackedIdle + fT * 1.2 + fA * 0.95;
          bb = dimTrackedIdle + fT * 1.92 + fA * 0.42;
          rr = Math.min(rr, cap);
          gg = Math.min(gg, cap);
          bb = Math.min(bb, cap);
        }
        const nearHead = Math.max(fT, fA);
        const headGate = 1 - Math.exp(-nearHead * 16);
        rr = dimUnrelated + headGate * (rr - dimUnrelated);
        gg = dimUnrelated + headGate * (gg - dimUnrelated);
        bb = dimUnrelated + headGate * (bb - dimUnrelated);
        if (inTgt) {
          const phas = 0.5 + 0.5 * Math.sin(uTime * 2.65);
          const amp = lightBg ? 0.07 : 0.17;
          const pulse = 1 + amp * fT * phas;
          const uplR = rr - dimUnrelated;
          const uplG = gg - dimUnrelated;
          const uplB = bb - dimUnrelated;
          rr = dimUnrelated + uplR * pulse;
          gg = dimUnrelated + uplG * pulse;
          bb = dimUnrelated + uplB * pulse;
          const capT = lightBg ? 0.52 : cap;
          rr = Math.min(rr, capT);
          gg = Math.min(gg, capT);
          bb = Math.min(bb, capT);
        }
        ar[o] = rr;
        ar[o + 1] = gg;
        ar[o + 2] = bb;
        if (Number.isFinite(obj.userData.corpusBaseScale) && (inTgt || inSrc)) {
          const bs = obj.userData.corpusBaseScale;
          let sizeMul = 1;
          if (inSrc) {
            const dxs = px - headAx;
            const dys = py - headAy;
            const dzs = pz - headAz;
            const fA_size = Math.exp(-(dxs * dxs + dys * dys + dzs * dzs) * invR2Size);
            sizeMul = 1 + fA_size * 4;
          }
          const spinY = inTgt ? fT * fT * headGate * uTime * 4.2 : 0;
          dummy.position.set(px, py, pz);
          dummy.rotation.set(0, spinY, 0);
          dummy.scale.setScalar(bs * sizeMul);
          dummy.updateMatrix();
          obj.setMatrixAt(j, dummy.matrix);
        }
      }
      obj.instanceColor.needsUpdate = true;
      if ((inTgt || inSrc) && Number.isFinite(obj.userData.corpusBaseScale)) {
        obj.instanceMatrix.needsUpdate = true;
      }
    });
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = isErr ? 'err' : 'ok';
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  function applyWhiteBackgroundFromUI() {
    const el = document.getElementById('whiteBgCb');
    const on = !!(el && el.checked);
    if (on) {
      scene.background.setHex(0xffffff);
      document.documentElement.style.background = '#ffffff';
      document.body.style.background = '#ffffff';
    } else {
      scene.background.setHex(0x000000);
      document.documentElement.style.background = '#000000';
      document.body.style.background = '#000000';
    }
    const legWin = document.getElementById('umapLegendWin');
    if (legWin) legWin.classList.toggle('umap-legend-light', on);
    const bar = document.getElementById('bar');
    if (bar) bar.classList.toggle('fa-bar-light', on);
    updatePathAnimation();
  }

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.02, 200);
  camera.position.set(2.8, 2.2, 2.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  let cloud = null;
  let lastMeta = null;
  let trajTube = null;
  let trajTubeHalo = null;
  let argTube = null;
  let headConnectorLine = null;
  let trajHull = null;
  let argHull = null;
  let trajPosF32 = null;
  let argPosF32 = null;
  let trajPosBaseF32 = null;
  let argPosBaseF32 = null;
  let pathVertexN = 0;
  let animStartMs = 0;
  let corpusProxWasOn = false;

  function decodeU32(b64, n) {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Uint32Array(raw.buffer, raw.byteOffset, n);
  }

  class PolylineCurve3 extends THREE.Curve {
    constructor(f32) {
      super();
      this.pts = [];
      for (let i = 0; i < f32.length / 3; i++) {
        this.pts.push(new THREE.Vector3(f32[i * 3], f32[i * 3 + 1], f32[i * 3 + 2]));
      }
    }
    getPoint(t, optionalTarget = new THREE.Vector3()) {
      const n = this.pts.length;
      if (n === 0) return optionalTarget.set(0, 0, 0);
      if (n === 1) return optionalTarget.copy(this.pts[0]);
      t = Math.min(1, Math.max(0, t));
      const f = t * (n - 1);
      const i0 = Math.floor(f);
      const i1 = Math.min(n - 1, i0 + 1);
      const w = f - i0;
      return optionalTarget.copy(this.pts[i0]).lerp(this.pts[i1], w);
    }
    getTangent(t, optionalTarget = new THREE.Vector3()) {
      const n = this.pts.length;
      const eps = 1 / Math.max(20, (n - 1) * 64);
      const t0 = Math.max(0, t - eps);
      const t1 = Math.min(1, t + eps);
      return optionalTarget.subVectors(this.getPoint(t1), this.getPoint(t0)).normalize();
    }
  }

  function cumulativeArcLength(f32) {
    const n = f32.length / 3;
    const cum = new Float32Array(n);
    let s = 0;
    cum[0] = 0;
    for (let i = 1; i < n; i++) {
      const o = i * 3;
      const p = (i - 1) * 3;
      const dx = f32[o] - f32[p];
      const dy = f32[o + 1] - f32[p + 1];
      const dz = f32[o + 2] - f32[p + 2];
      s += Math.sqrt(dx * dx + dy * dy + dz * dz);
      cum[i] = s;
    }
    return cum;
  }

  const GLASS_VS = `
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      vUv = uv;
      vNormal = normalize(normalMatrix * normal);
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  const GLASS_FS = `
    uniform float uHeadU;
    uniform vec3 uAccent;
    uniform float uFadeOn;
    uniform float uBand;
    uniform float uPulseSharp;
    uniform float uAlphaMul;
    uniform float uFlowHaze;
    uniform float uTime;
    uniform float uCorpusLightMode;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      float u = clamp(vUv.x, 0.0, 1.0);
      float cm = clamp(uCorpusLightMode, 0.0, 1.0);
      float tubeDim = mix(1.0, 0.09, cm);
      float hazeDim = mix(1.0, 0.18, cm);
      float pulse = exp(-pow((u - uHeadU) * uPulseSharp, 2.0));
      float tailMask = 1.0;
      if (uFadeOn > 0.5 && u < uHeadU - 1e-5) {
        float age = uHeadU - u;
        tailMask = exp(-age / max(uBand, 0.028));
      }
      vec3 viewDir = normalize(cameraPosition - vWorldPos);
      vec3 N = normalize(vNormal);
      float ndv = max(dot(N, viewDir), 0.0);
      float fresnel = pow(1.0 - ndv, 2.65);
      vec3 darkGlass = vec3(0.03, 0.06, 0.1);
      vec3 tint = mix(darkGlass, uAccent, 0.48);
      float headLight = pulse * (0.65 + 1.85 * tailMask) * tubeDim;
      vec3 core = uAccent * headLight;
      vec3 col = tint * (0.18 + 0.72 * fresnel) + core;
      float flow = u * 6.2831853 * 2.85 + uTime * 0.85;
      float hazeWave = 0.5 + 0.5 * sin(flow);
      float hazeEnvelope = (0.52 + 0.48 * tailMask);
      float veil = uFlowHaze * (0.11 + 0.46 * fresnel) * (0.48 + 0.52 * hazeWave) * hazeEnvelope * hazeDim;
      col += uAccent * veil * 0.48;
      float baseAlpha = 0.1 + 0.38 * fresnel + 0.58 * headLight;
      float alpha = clamp((baseAlpha + veil * 0.62) * uAlphaMul, 0.0, 1.0);
      gl_FragColor = vec4(col, alpha);
    }
  `;

  const HALO_FS = `
    uniform float uHeadU;
    uniform vec3 uAccent;
    uniform float uPulseSharp;
    uniform float uAlphaMul;
    uniform float uCorpusLightMode;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      float u = clamp(vUv.x, 0.0, 1.0);
      float cm = clamp(uCorpusLightMode, 0.0, 1.0);
      float tubeDim = mix(1.0, 0.14, cm);
      float pulse = exp(-pow((u - uHeadU) * uPulseSharp * 0.72, 2.0));
      vec3 viewDir = normalize(cameraPosition - vWorldPos);
      vec3 N = normalize(vNormal);
      float ndv = max(dot(N, viewDir), 0.0);
      float fresnel = pow(1.0 - ndv, 1.85);
      float head = pulse * (0.22 + 0.95 * tubeDim);
      vec3 col = uAccent * (0.06 * fresnel + head * 0.42);
      float alpha = clamp((0.06 + 0.38 * fresnel + head * 0.48) * uAlphaMul, 0.0, 0.5);
      gl_FragColor = vec4(col, alpha);
    }
  `;

  function makeGlassTube(posF32, accentColor, radius) {
    const n = posF32.length / 3;
    if (n < 2) return null;
    const curve = new PolylineCurve3(posF32);
    const segs = Math.min(720, Math.max(24, (n - 1) * 6));
    const tubeGeo = new THREE.TubeGeometry(curve, segs, radius, 10, false);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uHeadU: { value: 0 },
        uAccent: { value: accentColor },
        uFadeOn: { value: 1 },
        uBand: { value: 0.12 },
        uPulseSharp: { value: 52 },
        uAlphaMul: { value: 0 },
        uFlowHaze: { value: 1 },
        uTime: { value: 0 },
        uCorpusLightMode: { value: 0 },
      },
      vertexShader: GLASS_VS,
      fragmentShader: GLASS_FS,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const mesh = new THREE.Mesh(tubeGeo, mat);
    mesh.renderOrder = 4;
    mesh.userData.cum = cumulativeArcLength(posF32);
    mesh.userData.n = n;
    return mesh;
  }

  function makeTrajHaloTube(posF32, accentColor, baseRadius) {
    const n = posF32.length / 3;
    if (n < 2) return null;
    const curve = new PolylineCurve3(posF32);
    const segs = Math.min(720, Math.max(24, (n - 1) * 6));
    const haloR = baseRadius * 2.38;
    const tubeGeo = new THREE.TubeGeometry(curve, segs, haloR, 10, false);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uHeadU: { value: 0 },
        uAccent: { value: accentColor },
        uPulseSharp: { value: 52 },
        uAlphaMul: { value: 0.68 },
        uCorpusLightMode: { value: 0 },
      },
      vertexShader: GLASS_VS,
      fragmentShader: HALO_FS,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const mesh = new THREE.Mesh(tubeGeo, mat);
    mesh.renderOrder = 3;
    mesh.userData.cum = cumulativeArcLength(posF32);
    mesh.userData.n = n;
    return mesh;
  }

  function makeTightHullGroup(hullMeta, fillHex, edgeHex) {
    if (!hullMeta || !hullMeta.pos_b64 || !hullMeta.idx_b64) return null;
    const vcount = hullMeta.vcount | 0;
    const fcount = hullMeta.fcount | 0;
    if (vcount < 4 || fcount < 1) return null;
    const pos = decodeF32(hullMeta.pos_b64, vcount * 3);
    const idx = decodeU32(hullMeta.idx_b64, fcount * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(new THREE.Uint32BufferAttribute(idx, 1));
    g.computeVertexNormals();
    const mat = new THREE.MeshPhysicalMaterial({
      color: fillHex,
      metalness: 0.02,
      roughness: 0.06,
      transmission: 0.58,
      thickness: 0.62,
      ior: 1.47,
      transparent: true,
      opacity: 1.0,
      side: THREE.DoubleSide,
      depthWrite: false,
      attenuationColor: new THREE.Color(fillHex),
      attenuationDistance: 0.75,
      envMapIntensity: 1.0,
    });
    const mesh = new THREE.Mesh(g, mat);
    mesh.renderOrder = 1;
    const eg = new THREE.EdgesGeometry(g, 26);
    const em = new THREE.LineBasicMaterial({ color: edgeHex, transparent: true, opacity: 0.32, depthWrite: false });
    const lines = new THREE.LineSegments(eg, em);
    lines.renderOrder = 2;
    const grp = new THREE.Group();
    grp.add(mesh);
    grp.add(lines);
    grp.renderOrder = 1;
    return grp;
  }

  function disposeObj(o) {
    if (!o) return;
    scene.remove(o);
    o.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
          if (m.map) m.map.dispose();
          m.dispose();
        }
      }
    });
  }

  function disposeHeadLinkLines() {
    disposeObj(headConnectorLine);
    headConnectorLine = null;
  }

  function syncHeadLinkLinesVisibility() {
    const tr = document.getElementById('showTr');
    const am = document.getElementById('showAm');
    const vis = !!(tr && tr.checked && am && am.checked);
    if (headConnectorLine) headConnectorLine.visible = vis;
  }

  function ensureHeadLinkLines() {
    if (pathVertexN < 2 || !trajPosF32 || !argPosF32) return;
    if (headConnectorLine) return;
    const cbuf = new Float32Array(6);
    const cgeo = new THREE.BufferGeometry();
    cgeo.setAttribute('position', new THREE.BufferAttribute(cbuf, 3).setUsage(THREE.DynamicDrawUsage));
    headConnectorLine = new THREE.Line(
      cgeo,
      new THREE.LineBasicMaterial({
        color: 0xd63031,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      })
    );
    headConnectorLine.renderOrder = 6;
    scene.add(headConnectorLine);
    syncHeadLinkLinesVisibility();
  }

  function cbGrainRotation(i) {
    let s = (i * 1597334677) | 0;
    const u = (x) => ((x >>> 0) & 0xfffffff) / 0xfffffff;
    const rx = u(s) * Math.PI * 2;
    s = (s * 2246822519) | 0;
    const ry = u(s) * Math.PI * 2;
    s = (s * 2246822519) | 0;
    const rz = u(s) * Math.PI * 2;
    return [rx, ry, rz];
  }

  function buildCodebookCloud(C) {
    const n = C.count | 0;
    const cpos = decodeF32(C.pos_b64, n * 3);
    const crgb = decodeF32(C.rgb_b64, n * 3);
    const style = 'points';

    if (n < 1) {
      const empty = new THREE.Group();
      empty.renderOrder = 1;
      return empty;
    }

    if (style === 'points') {
      const cgeom = new THREE.BufferGeometry();
      cgeom.setAttribute('position', new THREE.Float32BufferAttribute(cpos, 3));
      cgeom.setAttribute('color', new THREE.Float32BufferAttribute(crgb, 3));
      const pts = new THREE.Points(
        cgeom,
        new THREE.PointsMaterial({
          size: 0.04,
          vertexColors: true,
          transparent: true,
          opacity: 0.75,
          sizeAttenuation: true,
        }),
      );
      pts.renderOrder = 1;
      return pts;
    }

    if (style === 'envelope') {
      const th = C.tight_hull;
      if (th && th.vcount && th.fcount) {
        const pre = makeTightHullGroup(th, 0x64748b, 0x94a3b8);
        if (pre) return pre;
      }
      const MAX_HULL = 3200;
      const step = n <= MAX_HULL ? 1 : Math.ceil(n / MAX_HULL);
      const vecs = [];
      for (let i = 0; i < n; i += step) {
        const o = i * 3;
        vecs.push(new THREE.Vector3(cpos[o], cpos[o + 1], cpos[o + 2]));
      }
      let sr = 0;
      let sg = 0;
      let sb = 0;
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        sr += crgb[o];
        sg += crgb[o + 1];
        sb += crgb[o + 2];
      }
      const inv = 1 / n;
      const meanCol = new THREE.Color(sr * inv, sg * inv, sb * inv);
      const slate = new THREE.Color(0x475569);
      const tint = slate.lerp(meanCol, 0.5);

      const group = new THREE.Group();
      group.renderOrder = 1;
      try {
        if (vecs.length < 4) throw new Error('need at least 4 points');
        const hullGeo = new ConvexGeometry(vecs);
        const shellMat = new THREE.MeshStandardMaterial({
          color: tint,
          metalness: 0.12,
          roughness: 0.42,
          transparent: true,
          opacity: 0.24,
          side: THREE.DoubleSide,
          depthWrite: false,
          envMapIntensity: 1.0,
        });
        const shell = new THREE.Mesh(hullGeo, shellMat);
        shell.renderOrder = 1;
        const edgeGeo = new THREE.EdgesGeometry(hullGeo, 42);
        const edgesMat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          transparent: true,
          opacity: 0.42,
          depthWrite: false,
        });
        const edges = new THREE.LineSegments(edgeGeo, edgesMat);
        edges.renderOrder = 2;
        group.add(shell);
        group.add(edges);
      } catch (e) {
        const cgeom = new THREE.BufferGeometry();
        cgeom.setAttribute('position', new THREE.Float32BufferAttribute(cpos, 3));
        cgeom.setAttribute('color', new THREE.Float32BufferAttribute(crgb, 3));
        const pts = new THREE.Points(
          cgeom,
          new THREE.PointsMaterial({
            size: 0.045,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            sizeAttenuation: true,
          }),
        );
        pts.renderOrder = 1;
        group.add(pts);
      }
      return group;
    }

    const specs = [
      {
        name: 'box',
        geo: new THREE.BoxGeometry(1, 1, 1),
        mat: new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          transparent: true,
          opacity: 0.88,
          metalness: 0.82,
          roughness: 0.22,
          envMapIntensity: 1.0,
        }),
        scale: 0.032,
      },
      {
        name: 'oct',
        geo: new THREE.OctahedronGeometry(0.95, 0),
        mat: new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          transparent: true,
          opacity: 0.86,
          metalness: 0.08,
          roughness: 0.72,
          envMapIntensity: 0.95,
        }),
        scale: 0.036,
      },
      {
        name: 'tetra',
        geo: new THREE.TetrahedronGeometry(1.02, 0),
        mat: new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          transparent: true,
          opacity: 0.87,
          metalness: 0.38,
          roughness: 0.28,
          clearcoat: 0.55,
          clearcoatRoughness: 0.18,
          envMapIntensity: 1.0,
        }),
        scale: 0.034,
      },
      {
        name: 'icosa',
        geo: new THREE.IcosahedronGeometry(0.72, 0),
        mat: new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          transparent: true,
          opacity: 0.9,
          metalness: 0.92,
          roughness: 0.12,
          envMapIntensity: 1.05,
        }),
        scale: 0.031,
      },
    ];

    const buckets = [[], [], [], []];
    for (let i = 0; i < n; i++) buckets[i % 4].push(i);

    const group = new THREE.Group();
    group.renderOrder = 1;
    const dummy = new THREE.Object3D();

    for (let k = 0; k < 4; k++) {
      const idxs = buckets[k];
      if (!idxs.length) {
        specs[k].geo.dispose();
        specs[k].mat.dispose();
        continue;
      }
      const { geo, mat, scale } = specs[k];
      const inst = new THREE.InstancedMesh(geo, mat, idxs.length);
      inst.instanceMatrix.setUsage(THREE.StaticDrawUsage);
      inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(idxs.length * 3), 3);

      for (let j = 0; j < idxs.length; j++) {
        const i = idxs[j];
        const o = i * 3;
        dummy.position.set(cpos[o], cpos[o + 1], cpos[o + 2]);
        const [rx, ry, rz] = cbGrainRotation(i);
        dummy.rotation.set(rx, ry, rz);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();
        inst.setMatrixAt(j, dummy.matrix);
        inst.instanceColor.setXYZ(j, crgb[o], crgb[o + 1], crgb[o + 2]);
      }
      inst.instanceMatrix.needsUpdate = true;
      inst.instanceColor.needsUpdate = true;
      inst.frustumCulled = true;
      group.add(inst);
    }
    return group;
  }

  function getTrajRadiiFromUI() {
    const elT = document.getElementById('trajRadTarget');
    const elA = document.getElementById('trajRadArgmax');
    const rT = elT ? parseFloat(elT.value) : NaN;
    const rA = elA ? parseFloat(elA.value) : NaN;
    return [
      Number.isFinite(rT) ? rT : 0.002,
      Number.isFinite(rA) ? rA : 0.022,
    ];
  }

  function blocksForSpace(META, key) {
    if (META.spaces && META.spaces[key]) {
      return META.spaces[key];
    }
    if (key === 'joint_pca' && META.codebook && META.target_traj && META.argmax_traj) {
      return { codebook: META.codebook, target_traj: META.target_traj, argmax_traj: META.argmax_traj };
    }
    return null;
  }

  function refitEmbedSpaceSelect(META) {
    const sel = document.getElementById('embedSpace');
    const row = document.getElementById('embedSpaceRow');
    let avail = [];
    if (Array.isArray(META.spaces_available) && META.spaces_available.length) {
      avail = META.spaces_available.slice();
    } else if (META.spaces && typeof META.spaces === 'object') {
      avail = Object.keys(META.spaces);
    } else {
      avail = ['joint_pca'];
    }
    avail = [...new Set(avail)];
    const order = ['joint_pca', 'global_umap', 'global_pca'];
    avail.sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      const sa = ia === -1 ? 999 : ia;
      const sb = ib === -1 ? 999 : ib;
      if (sa !== sb) return sa - sb;
      return a.localeCompare(b);
    });
    const labels = {
      joint_pca: 'Joint PCA (pair)',
      global_pca: 'Global PCA (corpus)',
      global_umap: 'Global UMAP (corpus)',
    };
    const prev = sel.value;
    sel.innerHTML = '';
    for (const k of avail) {
      const o = document.createElement('option');
      o.value = k;
      o.textContent = labels[k] || k;
      sel.appendChild(o);
    }
    if (avail.includes('global_umap')) {
      sel.value = 'global_umap';
    } else if (avail.includes('global_pca')) {
      sel.value = 'global_pca';
    } else if (avail.includes(prev)) {
      sel.value = prev;
    } else {
      sel.value = avail[0] || 'joint_pca';
    }
    row.style.display = avail.length > 1 ? '' : 'none';
  }

  function trajSpreadFactor() {
    const k = document.getElementById('embedSpace').value;
    if (k !== 'global_pca' && k !== 'global_umap') return 1;
    return getCorpusVizFromUI().spread;
  }

  function applyCorpusSpreadToPaths() {
    const key = document.getElementById('embedSpace').value;
    if (key !== 'global_pca' && key !== 'global_umap') return;
    if (!trajPosBaseF32 || !argPosBaseF32 || !trajPosF32 || !argPosF32) return;
    const s = trajSpreadFactor();
    for (let i = 0; i < trajPosBaseF32.length; i++) trajPosF32[i] = trajPosBaseF32[i] * s;
    for (let i = 0; i < argPosBaseF32.length; i++) argPosF32[i] = argPosBaseF32[i] * s;
    if (trajHull) trajHull.scale.setScalar(s);
    if (argHull) argHull.scale.setScalar(s);
    rebuildTrajectoryTubesOnly();
  }

  async function applyEmbeddingSpace(key) {
    if (!lastMeta) return;
    const B = blocksForSpace(lastMeta, key);
    if (!B) {
      setStatus('Missing embedding space: ' + key, true);
      return;
    }
    disposeObj(cloud);
    disposeObj(trajTube);
    disposeObj(trajTubeHalo);
    disposeObj(argTube);
    disposeHeadLinkLines();
    disposeObj(trajHull);
    disposeObj(argHull);
    cloud = null;
    trajTube = null;
    trajTubeHalo = null;
    argTube = null;
    trajHull = null;
    argHull = null;
    trajPosBaseF32 = null;
    argPosBaseF32 = null;

    if (key !== 'global_umap') hideUmapLegend();

    const corpusRow = document.getElementById('corpusFoldersRow');
    const corpusVizRow = document.getElementById('corpusVizRow');
    const useGlobalCorpus =
      (key === 'global_pca' || key === 'global_umap')
      && manifestRef
      && manifestRef.global_corpus_cloud;
    corpusRow.style.display = useGlobalCorpus ? '' : 'none';
    corpusVizRow.style.display = useGlobalCorpus ? '' : 'none';

    if (
      (key === 'global_pca' || key === 'global_umap')
      && manifestRef
      && manifestRef.global_corpus_cloud
    ) {
      try {
        const data = await fetchCorpusCloudJson(manifestRef.global_corpus_cloud);
        refitCorpusFolderOptions(data.folders);
        const blk = data.spaces && data.spaces[key];
        const filt = getCorpusFolderFilterSet();
        if (blk && Array.isArray(data.folder_styles) && data.folder_styles.length) {
          cloud = buildCorpusCloudGroup(blk, data.folder_styles, filt);
          if (key === 'global_umap') refitUmapLegend(data.folders, data.folder_styles);
          else hideUmapLegend();
        } else {
          cloud = buildCodebookCloud(B.codebook);
          hideUmapLegend();
        }
      } catch (e) {
        console.warn(e);
        cloud = buildCodebookCloud(B.codebook);
        hideUmapLegend();
      }
    } else {
      cloud = buildCodebookCloud(B.codebook);
      hideUmapLegend();
    }
    cloud.visible = true;
    scene.add(cloud);

    const T = B.target_traj;
    const tpos = decodeF32(T.pos_b64, T.count * 3);
    const A = B.argmax_traj;
    const apos = decodeF32(A.pos_b64, A.count * 3);
    trajPosBaseF32 = new Float32Array(tpos);
    argPosBaseF32 = new Float32Array(apos);
    const sSpread = trajSpreadFactor();
    trajPosF32 = new Float32Array(trajPosBaseF32.length);
    argPosF32 = new Float32Array(argPosBaseF32.length);
    for (let ti = 0; ti < trajPosBaseF32.length; ti++) trajPosF32[ti] = trajPosBaseF32[ti] * sSpread;
    for (let ti = 0; ti < argPosBaseF32.length; ti++) argPosF32[ti] = argPosBaseF32[ti] * sSpread;

    trajHull = makeTightHullGroup(T.tight_hull, 0x38bdf8, 0x7dd3fc);
    if (trajHull) {
      trajHull.scale.setScalar(sSpread);
      trajHull.visible = document.getElementById('showTr').checked;
      scene.add(trajHull);
    }
    argHull = makeTightHullGroup(A.tight_hull, 0xfb923c, 0xfbbf77);
    if (argHull) {
      argHull.scale.setScalar(sSpread);
      argHull.visible = document.getElementById('showAm').checked;
      scene.add(argHull);
    }

    const [rT, rA] = getTrajRadiiFromUI();
    trajTube = makeGlassTube(trajPosF32, new THREE.Color(0x56d4ff), rT);
    trajTubeHalo = makeTrajHaloTube(trajPosF32, new THREE.Color(0x56d4ff), rT);
    const trVis = document.getElementById('showTr').checked;
    if (trajTube) {
      trajTube.visible = trVis;
      scene.add(trajTube);
    }
    if (trajTubeHalo) {
      trajTubeHalo.visible = trVis;
      scene.add(trajTubeHalo);
    }
    argTube = makeGlassTube(argPosF32, new THREE.Color(0xffa066), rA);
    if (argTube) {
      argTube.visible = document.getElementById('showAm').checked;
      scene.add(argTube);
    }

    pathVertexN = T.count;
    animStartMs = performance.now();
    updatePathAnimation();
  }

  async function applyBundle(META) {
    document.getElementById('docTitle').textContent = META.title;
    document.getElementById('hdrTitle').textContent = META.title;

    lastMeta = META;
    refitEmbedSpaceSelect(META);
    const sel = document.getElementById('embedSpace');
    if (blocksForSpace(META, 'global_umap')) sel.value = 'global_umap';
    else if (blocksForSpace(META, 'global_pca')) sel.value = 'global_pca';
    await applyEmbeddingSpace(sel.value);
  }

  function updatePathAnimation() {
    const proxCb = document.getElementById('trajProxUmap');
    const corpusLight = proxCb && proxCb.checked ? 1.0 : 0.0;
    const pulseK = parseFloat(document.getElementById('trajPulse').value);
    const alphaM = parseFloat(document.getElementById('trajAlpha').value);
    const hazeEl = document.getElementById('trajHaze');
    const hazeV = hazeEl ? parseFloat(hazeEl.value) : NaN;
    const hazeMul = Number.isFinite(hazeV) ? hazeV : 1.0;
    const uTime = performance.now() * 0.001;
    for (const mesh of [trajTube, argTube, trajTubeHalo]) {
      if (mesh && mesh.material && mesh.material.uniforms) {
        const u = mesh.material.uniforms;
        if (u.uCorpusLightMode) {
          u.uCorpusLightMode.value = corpusLight;
        }
        if (Number.isFinite(pulseK) && u.uPulseSharp) {
          u.uPulseSharp.value = pulseK;
        }
        if (Number.isFinite(alphaM) && u.uAlphaMul) {
          u.uAlphaMul.value = mesh === trajTubeHalo ? Math.max(0.05, alphaM * 0.78) : alphaM;
        }
        if (u.uFlowHaze) {
          u.uFlowHaze.value = Math.max(0, hazeMul);
        }
        if (u.uTime) {
          u.uTime.value = uTime;
        }
      }
    }

    if (pathVertexN < 2 || !trajPosF32 || !argPosF32) {
      disposeHeadLinkLines();
      if (corpusProxWasOn) resetCorpusProximityColors();
      corpusProxWasOn = false;
      return;
    }
    const n = pathVertexN;
    const animOn = document.getElementById('animPath').checked;

    let floatAlong;
    if (!animOn) {
      floatAlong = Math.max(0, n - 1);
    } else {
      const periodSec = Math.max(0.5, parseFloat(document.getElementById('animPeriod').value) || 180);
      const prog = ((performance.now() - animStartMs) * 0.001 / periodSec) % 1;
      floatAlong = prog * Math.max(1, n - 1);
    }

    const nPos = Math.min(n, (trajPosF32.length / 3) | 0, (argPosF32.length / 3) | 0);
    const maxAlong = Math.max(0, nPos - 1);
    floatAlong = Math.min(maxAlong, Math.max(0, floatAlong));
    const i0 = Math.min(maxAlong, Math.floor(floatAlong));
    const i1 = Math.min(maxAlong, i0 + 1);
    const f = floatAlong - i0;

    const uBand = 0.12;
    const fadeUniform = 0.0;

    let trajHeadU = 0;
    if (trajTube && trajTube.userData.cum) {
      const cum = trajTube.userData.cum;
      const nn = trajTube.userData.n;
      const denom = Math.max(1e-8, cum[nn - 1]);
      const j0 = Math.min(nn - 1, i0);
      const j1 = Math.min(nn - 1, i1);
      const u0 = cum[j0] / denom;
      const u1 = cum[j1] / denom;
      trajHeadU = u0 + f * (u1 - u0);
      trajTube.material.uniforms.uHeadU.value = trajHeadU;
      trajTube.material.uniforms.uFadeOn.value = fadeUniform;
      trajTube.material.uniforms.uBand.value = uBand;
    }
    if (trajTubeHalo && trajTubeHalo.material.uniforms && trajTubeHalo.material.uniforms.uHeadU) {
      trajTubeHalo.material.uniforms.uHeadU.value = trajHeadU;
    }
    if (argTube && argTube.userData.cum) {
      const cum = argTube.userData.cum;
      const nn = argTube.userData.n;
      const denom = Math.max(1e-8, cum[nn - 1]);
      const j0 = Math.min(nn - 1, i0);
      const j1 = Math.min(nn - 1, i1);
      const u0 = cum[j0] / denom;
      const u1 = cum[j1] / denom;
      argTube.material.uniforms.uHeadU.value = u0 + f * (u1 - u0);
      argTube.material.uniforms.uFadeOn.value = fadeUniform;
      argTube.material.uniforms.uBand.value = uBand;
    }

    const o0 = i0 * 3;
    const o1 = i1 * 3;
    const tx = trajPosF32[o0] + f * (trajPosF32[o1] - trajPosF32[o0]);
    const ty = trajPosF32[o0 + 1] + f * (trajPosF32[o1 + 1] - trajPosF32[o0 + 1]);
    const tz = trajPosF32[o0 + 2] + f * (trajPosF32[o1 + 2] - trajPosF32[o0 + 2]);
    const ax = argPosF32[o0] + f * (argPosF32[o1] - argPosF32[o0]);
    const ay = argPosF32[o0 + 1] + f * (argPosF32[o1 + 1] - argPosF32[o0 + 1]);
    const az = argPosF32[o0 + 2] + f * (argPosF32[o1 + 2] - argPosF32[o0 + 2]);

    ensureHeadLinkLines();
    if (headConnectorLine) {
      syncHeadLinkLinesVisibility();
      const posC = headConnectorLine.geometry.attributes.position.array;
      posC[0] = tx;
      posC[1] = ty;
      posC[2] = tz;
      posC[3] = ax;
      posC[4] = ay;
      posC[5] = az;
      headConnectorLine.geometry.attributes.position.needsUpdate = true;
    }

    updateCorpusProximityGlow(tx, ty, tz, ax, ay, az);
  }

  function rebuildTrajectoryTubesOnly() {
    if (!trajPosF32 || !argPosF32 || pathVertexN < 2) return;
    const [rT, rA] = getTrajRadiiFromUI();
    disposeObj(trajTube);
    disposeObj(trajTubeHalo);
    disposeObj(argTube);
    trajTube = null;
    trajTubeHalo = null;
    argTube = null;
    trajTube = makeGlassTube(trajPosF32, new THREE.Color(0x56d4ff), rT);
    trajTubeHalo = makeTrajHaloTube(trajPosF32, new THREE.Color(0x56d4ff), rT);
    const trVis = document.getElementById('showTr').checked;
    if (trajTube) {
      trajTube.visible = trVis;
      scene.add(trajTube);
    }
    if (trajTubeHalo) {
      trajTubeHalo.visible = trVis;
      scene.add(trajTubeHalo);
    }
    argTube = makeGlassTube(argPosF32, new THREE.Color(0xffa066), rA);
    if (argTube) {
      argTube.visible = document.getElementById('showAm').checked;
      scene.add(argTube);
    }
    animStartMs = performance.now();
    updatePathAnimation();
  }

  function syncTrajectoryLookLabels() {
    const rt = document.getElementById('trajRadTarget');
    const ra = document.getElementById('trajRadArgmax');
    const rp = document.getElementById('trajPulse');
    const rx = document.getElementById('trajAlpha');
    const rh = document.getElementById('trajHaze');
    const vrt = document.getElementById('trajRadTargetVal');
    const vra = document.getElementById('trajRadArgmaxVal');
    const vp = document.getElementById('trajPulseVal');
    const vx = document.getElementById('trajAlphaVal');
    const vh = document.getElementById('trajHazeVal');
    const vprox = document.getElementById('trajProxRadiusVal');
    const rprox = document.getElementById('trajProxRadius');
    const vproxSize = document.getElementById('trajProxSizeRadiusVal');
    const rproxSize = document.getElementById('trajProxSizeRadius');
    if (rt && vrt) vrt.textContent = parseFloat(rt.value).toFixed(3);
    if (ra && vra) vra.textContent = parseFloat(ra.value).toFixed(3);
    if (rp && vp) vp.textContent = rp.value;
    if (rx && vx) vx.textContent = parseFloat(rx.value).toFixed(2);
    if (rh && vh) vh.textContent = parseFloat(rh.value).toFixed(2);
    if (rprox && vprox) vprox.textContent = parseFloat(rprox.value).toFixed(2);
    if (rproxSize && vproxSize) vproxSize.textContent = parseFloat(rproxSize.value).toFixed(2);
  }

  function wireTrajectoryLookControls() {
    const rt = document.getElementById('trajRadTarget');
    const ra = document.getElementById('trajRadArgmax');
    const rp = document.getElementById('trajPulse');
    const rx = document.getElementById('trajAlpha');
    const rh = document.getElementById('trajHaze');
    if (!rt || !ra || !rp || !rx || !rh) return;
    rt.addEventListener('input', () => { syncTrajectoryLookLabels(); rebuildTrajectoryTubesOnly(); });
    ra.addEventListener('input', () => { syncTrajectoryLookLabels(); rebuildTrajectoryTubesOnly(); });
    rp.addEventListener('input', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    rx.addEventListener('input', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    rh.addEventListener('input', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    syncTrajectoryLookLabels();
  }

  function wireCorpusProxControls() {
    const proxU = document.getElementById('trajProxUmap');
    const proxR = document.getElementById('trajProxRadius');
    const proxSizeR = document.getElementById('trajProxSizeRadius');
    if (proxU) {
      proxU.addEventListener('change', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    }
    if (proxR) {
      proxR.addEventListener('input', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    }
    if (proxSizeR) {
      proxSizeR.addEventListener('input', () => { syncTrajectoryLookLabels(); updatePathAnimation(); });
    }
  }

  async function reapplyCorpusCloudGeometry() {
    const key = document.getElementById('embedSpace').value;
    if (key !== 'global_pca' && key !== 'global_umap') return;
    if (!manifestRef || !manifestRef.global_corpus_cloud) return;
    try {
      const data = await fetchCorpusCloudJson(manifestRef.global_corpus_cloud);
      const blk = data.spaces && data.spaces[key];
      if (!blk || !Array.isArray(data.folder_styles) || !data.folder_styles.length) return;
      const filt = getCorpusFolderFilterSet();
      disposeObj(cloud);
      cloud = buildCorpusCloudGroup(blk, data.folder_styles, filt);
      cloud.visible = true;
      scene.add(cloud);
      if (key === 'global_umap') refitUmapLegend(data.folders, data.folder_styles);
    } catch (e) {
      console.warn(e);
    }
  }

  function syncCorpusVizLabels() {
    const op = document.getElementById('corpusOpacity');
    const ps = document.getElementById('corpusPointSize');
    const sp = document.getElementById('corpusSpread');
    const ov = document.getElementById('corpusOpacityVal');
    const psv = document.getElementById('corpusPointSizeVal');
    const spv = document.getElementById('corpusSpreadVal');
    if (op && ov) ov.textContent = parseFloat(op.value).toFixed(2);
    if (ps && psv) psv.textContent = parseFloat(ps.value).toFixed(3);
    if (sp && spv) spv.textContent = parseFloat(sp.value).toFixed(2);
  }

  function wireCorpusVizControls() {
    const op = document.getElementById('corpusOpacity');
    const ps = document.getElementById('corpusPointSize');
    const sp = document.getElementById('corpusSpread');
    if (!op || !ps || !sp) return;
    const rebuildCloud = () => {
      reapplyCorpusCloudGeometry().catch(() => {});
    };
    op.addEventListener('input', () => {
      syncCorpusVizLabels();
      rebuildCloud();
    });
    ps.addEventListener('input', () => {
      syncCorpusVizLabels();
      rebuildCloud();
    });
    sp.addEventListener('input', () => {
      syncCorpusVizLabels();
      rebuildCloud();
      applyCorpusSpreadToPaths();
    });
    syncCorpusVizLabels();
  }

  document.getElementById('embedSpace').addEventListener('change', () => {
    applyEmbeddingSpace(document.getElementById('embedSpace').value).catch((e) => {
      setStatus(String(e && e.message ? e.message : e), true);
    });
  });
  document.getElementById('showTr').addEventListener('change', (e) => {
    if (trajTube) trajTube.visible = e.target.checked;
    if (trajTubeHalo) trajTubeHalo.visible = e.target.checked;
    if (trajHull) trajHull.visible = e.target.checked;
    updatePathAnimation();
  });
  document.getElementById('showAm').addEventListener('change', (e) => {
    if (argTube) argTube.visible = e.target.checked;
    if (argHull) argHull.visible = e.target.checked;
    updatePathAnimation();
  });
  document.getElementById('animPath').addEventListener('change', () => { animStartMs = performance.now(); updatePathAnimation(); });
  const periodEl = document.getElementById('animPeriod');
  const periodVal = document.getElementById('animPeriodVal');
  periodEl.addEventListener('input', () => {
    periodVal.textContent = periodEl.value;
    animStartMs = performance.now();
    updatePathAnimation();
  });

  wireTrajectoryLookControls();
  wireCorpusProxControls();
  const whiteBg = document.getElementById('whiteBgCb');
  if (whiteBg) whiteBg.addEventListener('change', applyWhiteBackgroundFromUI);
  wireCorpusVizControls();

  const barEl = document.getElementById('bar');
  const barCollapse = document.getElementById('barCollapse');
  if (barEl && barCollapse) {
    barCollapse.addEventListener('click', () => {
      const collapsed = barEl.classList.toggle('bar-panel-collapsed');
      barCollapse.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      barCollapse.textContent = collapsed ? 'Expand' : 'Collapse';
    });
  }

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dl = new THREE.DirectionalLight(0xffffff, 0.8);
  dl.position.set(4, 6, 3);
  scene.add(dl);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function tick() {
    requestAnimationFrame(tick);
    updatePathAnimation();
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  function bundleUrl(source, tgtFolder, stem) {
    const id = cleanBundlePart(source) + '__' + cleanBundlePart(tgtFolder) + '__' + cleanBundlePart(stem);
    return 'bundles/' + id + '.json';
  }

  async function fetchBundleMeta(url) {
    if (bundleCache.has(url)) return bundleCache.get(url);
    if (bundleInflight.has(url)) return bundleInflight.get(url);
    const p = fetch(url, { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((meta) => {
        bundleCache.set(url, meta);
        bundleInflight.delete(url);
        return meta;
      })
      .catch((e) => {
        bundleInflight.delete(url);
        throw e;
      });
    bundleInflight.set(url, p);
    return p;
  }

  async function tryLoadBundle() {
    const source = document.getElementById('selSource').value;
    const tf = document.getElementById('selTargetFolder').value;
    const stem = document.getElementById('selTargetStem').value;
    if (!source || !tf || !stem) {
      setStatus('Select source folder and target sound.', true);
      return false;
    }
    const url = bundleUrl(source, tf, stem);
    try {
      if (bundleCache.has(url)) {
        const meta = bundleCache.get(url);
        await applyBundle(meta);
        setStatus('Loaded (cache) · ' + (meta.bundle_id || url), false);
        return true;
      }
      setStatus('Fetching ' + url + ' …', false);
      const META = await fetchBundleMeta(url);
      await applyBundle(META);
      setStatus('Loaded bundle: ' + (META.bundle_id || url), false);
      return true;
    } catch (e) {
      const code = e && e.message ? e.message : '';
      if (/^\d+$/.test(code)) {
        setStatus(
          'No bundle (HTTP ' + code + '). Precompute: --build-all-bundles or --source-folder + --target-npy',
          true,
        );
        return false;
      }
      setStatus('Fetch failed: ' + (e && e.message ? e.message : String(e)), true);
      return false;
    }
  }

  async function main() {
    let manifest;
    try {
      const r = await fetch(MANIFEST_URL, { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
      manifest = await r.json();
    } catch (e) {
      setStatus('Missing bundles/manifest.json — run the build script once without --target-npy, or use --manifest-only.', true);
      return;
    }

    const srcSel = document.getElementById('selSource');
    const tfSel = document.getElementById('selTargetFolder');
    const stemSel = document.getElementById('selTargetStem');

    for (const f of manifest.folders) {
      const o = document.createElement('option');
      o.value = f;
      o.textContent = f;
      srcSel.appendChild(o);
    }

    const byFolder = new Map();
    for (const t of manifest.targets) {
      if (!byFolder.has(t.folder)) byFolder.set(t.folder, []);
      byFolder.get(t.folder).push(t.stem);
    }
    for (const f of manifest.folders) {
      const o = document.createElement('option');
      o.value = f;
      o.textContent = f;
      tfSel.appendChild(o);
    }

    function refillStems() {
      const f = tfSel.value;
      stemSel.innerHTML = '';
      const stems = byFolder.get(f) || [];
      for (const s of stems) {
        const o = document.createElement('option');
        o.value = s;
        o.textContent = s;
        stemSel.appendChild(o);
      }
    }

    manifestRef = manifest;

    tfSel.addEventListener('change', () => {
      refillStems();
      tryLoadBundle();
    });
    srcSel.addEventListener('change', () => {
      tryLoadBundle();
    });
    stemSel.addEventListener('change', tryLoadBundle);

    refillStems();
    const folderArr = manifest.folders;
    let loaded = false;
    if (folderArr.length > 0) {
      const nF = folderArr.length;
      for (let attempt = 0; attempt < 28 && !loaded; attempt++) {
        srcSel.selectedIndex = Math.floor(Math.random() * nF);
        tfSel.selectedIndex = Math.floor(Math.random() * nF);
        refillStems();
        if (stemSel.options.length === 0) continue;
        stemSel.selectedIndex = Math.floor(Math.random() * stemSel.options.length);
        loaded = await tryLoadBundle();
      }
    }
    if (!loaded && manifest.bundles_available && manifest.bundles_available.length) {
      const bids = manifest.bundles_available;
      const pick = bids[Math.floor(Math.random() * bids.length)];
      const parts = pick.split('__');
      if (parts.length >= 3) {
        const stemVal = parts.slice(2).join('__');
        if (folderArr.includes(parts[0])) srcSel.value = parts[0];
        if (folderArr.includes(parts[1])) tfSel.value = parts[1];
        refillStems();
        if (Array.from(stemSel.options).some((o) => o.value === stemVal)) {
          stemSel.value = stemVal;
        } else if (stemSel.options.length > 0) {
          stemSel.selectedIndex = Math.floor(Math.random() * stemSel.options.length);
        }
        loaded = await tryLoadBundle();
      }
    }
    if (!loaded) {
      await tryLoadBundle();
    }
  }

  wireCorpusFolderFilterMount();

  main();
  </script>
</body>
</html>
"""


def write_viewer_html(out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(VIEWER_HTML)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enc-root", default="data/enc24k")
    p.add_argument("--out", default="results/latent_granular/latent_granular_3d_viewer.html")
    p.add_argument("--bundles-dir", default="", help="Default: <dirname(out)>/bundles")
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write viewer + scan enc-root into bundles/manifest.json (no bundle export)",
    )
    p.add_argument("--source-folder", default="", help="Codebook folder (required with --target-npy)")
    p.add_argument("--target-npy", default="", help="Path to target .npy under enc-root")
    p.add_argument("--source-glob", default="*.npy")
    p.add_argument("--max-source-files", type=int, default=None)
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--grain-mode", choices=("mean", "concat"), default="mean")
    p.add_argument("--assume-layout", choices=("row_is_time", "col_is_time"), default="row_is_time")
    p.add_argument("--max-codebook-points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-tight-hulls",
        action="store_true",
        help="Do not embed alpha-shape tight_hull meshes in bundles (smaller JSON, faster export)",
    )
    p.add_argument(
        "--hull-alpha-percentile",
        type=float,
        default=28.0,
        help="Alpha-ball percentile for 3D alpha shapes (lower ≈ tighter cavity, range ~5–95)",
    )
    p.add_argument(
        "--hull-cb-max-pts",
        type=int,
        default=3500,
        help="Max codebook points fed to Delaunay for the codebook hull",
    )
    p.add_argument(
        "--hull-traj-max-pts",
        type=int,
        default=4000,
        help="Max trajectory vertices fed to Delaunay for path envelopes",
    )
    p.add_argument(
        "--build-all-bundles",
        action="store_true",
        help="For every (enc folder as codebook) × (every .npy target), write bundles/*.json",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --build-all-bundles, do not overwrite existing bundle JSON",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N pairs (0 = silent)",
    )
    p.add_argument(
        "--global-embed-npz",
        default="",
        help=(
            "Optional PCA export from export_global_grain_embedding.py (.npz); "
            "fingerprint must match --window-size/--stride/--grain-mode"
        ),
    )
    p.add_argument(
        "--global-umap-model",
        default="",
        help="Optional fitted UMAP .joblib from the same export; requires --global-embed-npz",
    )
    args = p.parse_args()

    out_abs = os.path.abspath(args.out)
    bundles_dir = args.bundles_dir or os.path.join(os.path.dirname(out_abs), "bundles")
    os.makedirs(bundles_dir, exist_ok=True)

    write_viewer_html(out_abs)
    print(f"Wrote viewer {out_abs}")

    g_npz = (args.global_embed_npz or "").strip() or None
    g_umap = (args.global_umap_model or "").strip() or None

    if args.build_all_bundles:
        assume_row = args.assume_layout == "row_is_time"
        stride = args.stride if args.stride is not None else args.window_size
        th = not args.no_tight_hulls
        build_all_bundles(
            args.enc_root,
            bundles_dir,
            source_glob=args.source_glob,
            max_source_files=args.max_source_files,
            window_size=args.window_size,
            stride=stride,
            grain_mode=args.grain_mode,  # type: ignore[arg-type]
            assume_row_is_time=assume_row,
            max_codebook_points=args.max_codebook_points,
            seed=args.seed,
            skip_existing=args.skip_existing,
            progress_every=args.progress_every,
            tight_hulls=th,
            hull_cb_max_pts=args.hull_cb_max_pts,
            hull_traj_max_pts=args.hull_traj_max_pts,
            hull_alpha_percentile=args.hull_alpha_percentile,
            global_embed_npz=g_npz,
            global_umap_joblib=g_umap,
        )
        avail = discover_bundles(bundles_dir)
        gc_rel = resolve_global_corpus_cloud_manifest_url(g_npz, bundles_dir)
        mp = write_manifest(
            bundles_dir, args.enc_root, bundles_available=avail, global_corpus_cloud=gc_rel
        )
        print(f"Updated manifest {mp} ({len(avail)} bundle files)")
        return

    if args.manifest_only or not args.source_folder or not args.target_npy:
        gc_rel = resolve_global_corpus_cloud_manifest_url(g_npz, bundles_dir)
        mp = write_manifest(bundles_dir, args.enc_root, global_corpus_cloud=gc_rel)
        print(f"Wrote manifest {mp}")
        if args.manifest_only:
            return
        print("Tip: export a bundle per (source, target) pair:")
        print("  --source-folder instrument_samples --target-npy data/enc24k/soundscapes/....npy")
        return

    assume_row = args.assume_layout == "row_is_time"
    stride = args.stride if args.stride is not None else args.window_size
    th = not args.no_tight_hulls
    meta, bid = compute_bundle(
        args.enc_root,
        args.source_folder,
        args.target_npy,
        args.source_glob,
        args.max_source_files,
        args.window_size,
        stride,
        args.grain_mode,  # type: ignore[arg-type]
        assume_row,
        args.max_codebook_points,
        args.seed,
        tight_hulls=th,
        hull_cb_max_pts=args.hull_cb_max_pts,
        hull_traj_max_pts=args.hull_traj_max_pts,
        hull_alpha_percentile=args.hull_alpha_percentile,
        global_embed_npz=g_npz,
        global_umap_joblib=g_umap,
    )
    bundle_path = os.path.join(bundles_dir, bid + ".json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=0)
    print(f"Wrote bundle {bundle_path}")

    avail = discover_bundles(bundles_dir)
    gc_rel = resolve_global_corpus_cloud_manifest_url(g_npz, bundles_dir)
    mp = write_manifest(
        bundles_dir, args.enc_root, bundles_available=avail, global_corpus_cloud=gc_rel
    )
    print(f"Updated manifest {mp}")


if __name__ == "__main__":
    main()
