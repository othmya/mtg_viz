#!/usr/bin/env python3
"""
Build an interactive 3D HTML viewer (Three.js) with hulls, graph, points, and
time-ordered sound trajectories:

  1) Folder convex hulls — convex shells per enc24k folder (SciPy ConvexHull).
  2) Similarity graph — file centroids + kNN edges in 3D UMAP; optional reference
     sound tints each edge by min centroid distance (red = close, blue = far).
  3) Points — subsampled UMAP slice cloud; folder checkboxes filter which
     folders appear; click points/spheres/hulls for a metadata panel.
  4) Trajectories — per-sound polyline in global UMAP (time order); optional
     intrinsic PCA(3) per file when built with --intrinsic-pca.

Requires: scipy, scikit-learn. Serve the HTML over HTTP (ES modules), e.g.:
  cd results/PCA_plots && python3 -m http.server 8765
  → http://localhost:8765/enc24k_umap_structures_viewer.html

Prerequisite: run pca_enc24k_all.py so enc24k_umap_3d_export.npz exists.

Optional: rebuild with intrinsic trajectories (reads data/enc24k):
  python3 scripts/build_umap_structures_viewer.py --intrinsic-pca --intrinsic-max-sounds 64
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = 0.5 * (lo + hi)
    span = np.max(hi - lo)
    if span <= 1e-12:
        span = 1.0
    return (xyz - c) * (2.2 / span)


def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode(
        "ascii"
    )


def _b64_u32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint32).tobytes()).decode(
        "ascii"
    )


def _folder_colours(folder_names: list[str]) -> list[str]:
    # Distinct sRGB hex for UI + Three (parseInt 0x...)
    palette = [
        "#c41e3a",
        "#2e7d32",
        "#1565c0",
        "#6a1b9a",
        "#ef6c00",
        "#00838f",
        "#5d4037",
    ]
    return [palette[i % len(palette)] for i in range(len(folder_names))]


def _convex_hull_mesh(
    pts: np.ndarray,
    rng: np.random.Generator,
    max_input_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (positions Nx3 float32, triangle indices Mx3 uint32) or None."""
    if pts.shape[0] < 8:
        return None
    if pts.shape[0] > max_input_points:
        idx = rng.choice(pts.shape[0], size=max_input_points, replace=False)
        pts = pts[idx]
    p = np.asarray(pts, dtype=np.float64)
    try:
        hull = ConvexHull(p)
    except QhullError:
        p = p + rng.normal(0.0, 1e-5, size=p.shape)
        try:
            hull = ConvexHull(p)
        except QhullError:
            return None
    vertices = np.asarray(hull.points, dtype=np.float32)
    faces = np.asarray(hull.simplices, dtype=np.uint32)
    return vertices, faces


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        default="results/PCA_plots/enc24k_umap_3d_export.npz",
        help="UMAP export from pca_enc24k_all.py",
    )
    p.add_argument(
        "--out",
        default="results/PCA_plots/enc24k_umap_structures_viewer.html",
    )
    p.add_argument(
        "--enc-root",
        default="data/enc24k",
        help="Used for optional per-file intrinsic PCA trajectories",
    )
    p.add_argument(
        "--trajectory-max-points",
        type=int,
        default=8000,
        help="Max points per sound for trajectory polylines (both UMAP and intrinsic)",
    )
    p.add_argument(
        "--intrinsic-pca",
        action="store_true",
        help="Embed per-file PCA-3D trajectories (reads .npy under enc-root)",
    )
    p.add_argument(
        "--intrinsic-max-sounds",
        type=int,
        default=64,
        help="Cap how many sounds get intrinsic PCA (first IDs 0..N-1)",
    )
    p.add_argument(
        "--hull-max-points",
        type=int,
        default=2000,
        help="Max slice points per folder fed to ConvexHull",
    )
    p.add_argument(
        "--graph-k",
        type=int,
        default=4,
        help="k for kNN edges between file centroids",
    )
    p.add_argument(
        "--points-max",
        type=int,
        default=45_000,
        help="Max slice points embedded for the Points view (random subsample)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.npz)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    folder_id = np.asarray(data["folder_id"], dtype=np.int32)
    sound_id = np.asarray(data["sound_id"], dtype=np.int32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)

    folder_names = [str(x) for x in np.asarray(data["folders"]).tolist()]
    n_folders = len(folder_names)
    colours_hex = _folder_colours(folder_names)

    xyz_n = _normalize_xyz(xyz).astype(np.float32)
    rng = np.random.default_rng(args.seed)

    hulls: list[dict[str, Any]] = []
    for fi in range(n_folders):
        mask = folder_id == fi
        pts = xyz_n[mask]
        hm = _convex_hull_mesh(pts, rng, args.hull_max_points)
        if hm is None:
            continue
        vertices, faces = hm
        hulls.append(
            {
                "name": folder_names[fi],
                "pos_b64": _b64_f32(vertices.reshape(-1)),
                "idx_b64": _b64_u32(faces.reshape(-1)),
                "n_vertices": int(vertices.shape[0]),
                "n_triangles": int(faces.shape[0]),
                "color": colours_hex[fi],
            }
        )

    # File centroids in normalized space
    n_sounds = int(sound_id.max()) + 1 if sound_id.size else 0
    cents = np.zeros((n_sounds, 3), dtype=np.float64)
    cols = np.zeros((n_sounds, 3), dtype=np.float64)
    counts = np.zeros(n_sounds, dtype=np.int64)
    for i in range(xyz_n.shape[0]):
        sid = int(sound_id[i])
        cents[sid] += xyz_n[i]
        cols[sid] += rgb[i]
        counts[sid] += 1
    nz = counts > 0
    cents[nz] /= counts[nz, None]
    cols[nz] /= counts[nz, None]
    cents = cents.astype(np.float32)
    cols = np.clip(cols, 0.0, 1.0).astype(np.float32)

    k = min(args.graph_k, max(1, n_sounds - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(cents)
    _, ind = nn.kneighbors(cents)
    edges_set: set[tuple[int, int]] = set()
    for i in range(n_sounds):
        for j in ind[i, 1:]:  # skip self
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges_set.add((a, b))
    edges = np.array(list(edges_set), dtype=np.int32)

    sounds_arr = np.asarray(data["sounds"])
    folder_counts = [int((folder_id == i).sum()) for i in range(n_folders)]
    legend_folders = [
        {"name": folder_names[i], "color": colours_hex[i], "count": folder_counts[i]}
        for i in range(n_folders)
    ]
    legend_sounds = [
        {
            "label": str(sounds_arr[i]),
            "r": float(cols[i, 0]),
            "g": float(cols[i, 1]),
            "b": float(cols[i, 2]),
            "n_slices": int(counts[i]),
        }
        for i in range(n_sounds)
    ]

    total_n = xyz_n.shape[0]
    if total_n > args.points_max:
        pidx = rng.choice(total_n, size=args.points_max, replace=False)
    else:
        pidx = np.arange(total_n, dtype=np.int64)
    p_xyz = xyz_n[pidx]
    p_rgb = np.clip(rgb[pidx], 0.0, 1.0).astype(np.float32)
    p_fid = folder_id[pidx].astype(np.float32)
    p_sid = sound_id[pidx].astype(np.float32)
    inter_pts = np.empty((p_xyz.shape[0], 8), dtype=np.float32)
    inter_pts[:, :3] = p_xyz
    inter_pts[:, 3:6] = p_rgb
    inter_pts[:, 6] = p_fid
    inter_pts[:, 7] = p_sid

    sound_meta: list[dict[str, Any]] = []
    for i in range(n_sounds):
        lab = str(sounds_arr[i])
        parts = lab.split("/", 1)
        folder_name = parts[0] if len(parts) > 1 else ""
        stem = parts[1] if len(parts) > 1 else lab
        sound_meta.append(
            {
                "id": i,
                "label": lab,
                "folder": folder_name,
                "stem": stem,
                "n_slices": int(counts[i]),
            }
        )
    instrument_sound_ids = [s["id"] for s in sound_meta if s["folder"] == "instrument_samples"]

    tmax = args.trajectory_max_points
    trajectories_umap: list[dict[str, Any] | None] = []
    for sid in range(n_sounds):
        idx = np.flatnonzero(sound_id == sid)
        if idx.size < 2:
            trajectories_umap.append(None)
            continue
        if idx.size > tmax:
            pick = np.linspace(0, idx.size - 1, num=tmax, dtype=int)
            idx = idx[pick]
        pts = xyz_n[idx].astype(np.float32)
        trajectories_umap.append(
            {
                "n": int(pts.shape[0]),
                "pos_b64": _b64_f32(pts.reshape(-1)),
                "full_slices": int(counts[sid]),
            }
        )

    trajectories_intrinsic: list[dict[str, Any] | None] = [None] * n_sounds
    if args.intrinsic_pca:
        n_intr = min(args.intrinsic_max_sounds, n_sounds)
        for sid in range(n_intr):
            sm = sound_meta[sid]
            npy_path = os.path.join(args.enc_root, sm["folder"], sm["stem"] + ".npy")
            if not os.path.isfile(npy_path):
                continue
            z = np.load(npy_path)
            if z.dtype != np.float32:
                z = z.astype(np.float32, copy=False)
            if z.ndim != 2:
                continue
            if z.shape[0] == 128 and z.shape[1] != 128:
                z = z.T
            if z.shape[1] != 128 or z.shape[0] < 2:
                continue
            zp = z.astype(np.float64)
            kc = min(3, zp.shape[0], zp.shape[1])
            pca_m = PCA(n_components=kc, random_state=42)
            emb = pca_m.fit_transform(zp)
            if kc < 3:
                pad = np.zeros((emb.shape[0], 3), dtype=np.float64)
                pad[:, :kc] = emb
                emb = pad
            emb_n = _normalize_xyz(emb.astype(np.float32))
            if emb_n.shape[0] > tmax:
                pick = np.linspace(0, emb_n.shape[0] - 1, num=tmax, dtype=int)
                emb_n = emb_n[pick].astype(np.float32)
            trajectories_intrinsic[sid] = {
                "n": int(emb_n.shape[0]),
                "pos_b64": _b64_f32(emb_n.reshape(-1)),
                "npy_rel": os.path.join(sm["folder"], sm["stem"] + ".npy"),
            }

    meta: dict[str, Any] = {
        "hulls": hulls,
        "graph": {
            "n_nodes": n_sounds,
            "nodes_pos_b64": _b64_f32(cents.reshape(-1)),
            "nodes_rgb_b64": _b64_f32(cols.reshape(-1)),
            "edges": edges.tolist(),
        },
        "points": {
            "count": int(inter_pts.shape[0]),
            "stride": 8,
            "interp_b64": _b64_f32(inter_pts.reshape(-1)),
        },
        "folder_names": folder_names,
        "sound_meta": sound_meta,
        "instrument_sound_ids": instrument_sound_ids,
        "trajectories_umap": trajectories_umap,
        "trajectories_intrinsic": trajectories_intrinsic,
        "figure_notes": {
            "global_umap": (
                "3D coordinates are from UMAP fit on all pooled EnCodec slice rows (scripts/pca_enc24k_all.py). "
                "This is an extrinsic view of the corpus, not the model’s native Riemannian chart."
            ),
            "intrinsic_pca": (
                "PCA(3) is fit independently on each selected sound’s (T,128) matrix under data/enc24k—an intrinsic "
                "summary with no cross-file competition. Normalized to the same scale as hulls for visual context."
            ),
            "latent_granular": (
                "For cosine/softmax matching, codebook graphs, and Laplacian-style plots, run scripts/latent_granular_viz.py."
            ),
            "decode": (
                "Hearing the effect of moving along these curves requires EnCodec decode; this viewer only shows coordinates."
            ),
        },
        "legend": {
            "folders": legend_folders,
            "sounds": legend_sounds,
        },
    }

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UMAP 3D — hulls, graph & points</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #f8fafc; }
    #bar {
      position: fixed; top: 12px; left: 12px; z-index: 10;
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
      font: 14px system-ui, sans-serif; color: #0f172a;
    }
    #bar label { font-weight: 600; margin-right: 2px; }
    #viewSelect {
      cursor: pointer; border-radius: 10px; border: 1px solid #cbd5e1;
      padding: 8px 12px; background: rgba(255,255,255,0.92);
      backdrop-filter: blur(8px); font: inherit; min-width: 200px;
    }
    #hint {
      position: fixed; left: 12px; bottom: 10px; z-index: 10; max-width: 520px;
      font: 12px/1.4 system-ui, sans-serif; color: #334155;
      background: rgba(255,255,255,0.78); backdrop-filter: blur(10px);
      border: 1px solid rgba(148,163,184,0.4); border-radius: 10px; padding: 10px 12px;
    }
    #legendWrap {
      position: fixed; top: 12px; right: 12px; z-index: 11;
      font: 13px system-ui, sans-serif; color: #0f172a;
    }
    #legendWrap details {
      background: rgba(255,255,255,0.94); border: 1px solid #cbd5e1;
      border-radius: 10px; max-width: min(380px, calc(100vw - 24px));
      box-shadow: 0 4px 20px rgba(15,23,42,0.08);
    }
    #legendWrap summary {
      cursor: pointer; padding: 10px 14px; font-weight: 600; list-style: none;
      user-select: none;
    }
    #legendWrap summary::-webkit-details-marker { display: none; }
    #legendWrap .legend-panel {
      padding: 4px 12px 12px; max-height: min(72vh, 560px); overflow-y: auto;
      border-top: 1px solid #e2e8f0;
    }
    .legend-section {
      font-weight: 600; margin: 10px 0 6px; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; color: #64748b;
    }
    .legend-row {
      display: flex; align-items: flex-start; gap: 8px; margin: 5px 0;
      font-size: 12px; color: #334155; line-height: 1.35;
    }
    .legend-swatch {
      width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0;
      margin-top: 2px; border: 1px solid rgba(0,0,0,0.12);
    }
    .legend-meta { margin-left: auto; flex-shrink: 0; opacity: 0.65; font-variant-numeric: tabular-nums; }
    #sideControls {
      position: fixed; left: 12px; top: 52px; z-index: 10; width: 268px; max-height: 42vh;
      overflow-y: auto; font: 12px/1.35 system-ui, sans-serif; color: #0f172a;
      background: rgba(255,255,255,0.92); border: 1px solid #cbd5e1; border-radius: 10px;
      padding: 10px 12px; box-shadow: 0 4px 16px rgba(15,23,42,0.06);
    }
    #sideControls h3 { margin: 0 0 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
    #sideControls .block { margin-bottom: 12px; }
    #sideControls label.cb { display: flex; align-items: center; gap: 6px; margin: 4px 0; cursor: pointer; font-size: 12px; }
    #sideControls select { width: 100%; margin-top: 4px; padding: 6px 8px; border-radius: 8px; border: 1px solid #cbd5e1; font: inherit; }
    #sideControls input[type="range"] { width: 100%; margin-top: 2px; }
    #sideControls .traj-progress-label { display: block; font-size: 11px; color: #64748b; margin-top: 10px; }
    #sideControls .traj-progress-row { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
    #sideControls #trajProgressVal { font-variant-numeric: tabular-nums; color: #475569; font-size: 11px; min-width: 4.5rem; text-align: right; flex-shrink: 0; }
    #infoPanel {
      position: fixed; right: 12px; bottom: 12px; z-index: 12; width: min(320px, calc(100vw - 24px));
      max-height: 38vh; overflow-y: auto; font: 12px/1.45 system-ui, sans-serif; color: #0f172a;
      background: rgba(255,255,255,0.95); border: 1px solid #94a3b8; border-radius: 10px;
      padding: 10px 12px; box-shadow: 0 6px 24px rgba(15,23,42,0.12);
    }
    #infoPanel h3 { margin: 0 0 6px; font-size: 13px; }
    #infoPanel dl { margin: 0; }
    #infoPanel dt { color: #64748b; font-size: 10px; text-transform: uppercase; margin-top: 6px; }
    #infoPanel dd { margin: 0 0 2px 0; word-break: break-word; }
    #infoPanel .muted { color: #64748b; font-size: 11px; margin-top: 8px; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
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
    <label for="viewSelect">View</label>
    <select id="viewSelect" aria-label="Visualization mode">
      <option value="hull">Folder hulls</option>
      <option value="graph">File graph (kNN)</option>
      <option value="points">Points (slices)</option>
      <option value="traj_umap">Sound trajectory (UMAP)</option>
      <option value="traj_intrinsic">Sound trajectory (intrinsic PCA-3D)</option>
    </select>
  </div>
  <div id="legendWrap">
    <details id="legendDetails">
      <summary>Legend</summary>
      <div class="legend-panel" id="legendBody"></div>
    </details>
  </div>
  <aside id="sideControls">
    <div class="block">
      <h3>Point cloud — folders</h3>
      <div id="folderFilterMount"></div>
    </div>
    <div class="block">
      <h3>kNN edge heat (graph view)</h3>
      <div>Colour each edge by how close the <em>closer</em> endpoint is to this sound’s centroid in 3D UMAP (red = near, blue = far).</div>
      <label for="refSound" class="sr-only">Reference sound</label>
      <select id="refSound" aria-label="Reference sound for kNN edge colouring">
        <option value="">(none — neutral edges)</option>
      </select>
    </div>
    <div class="block">
      <h3>Trajectory — pick sound</h3>
      <div>Polyline = time-ordered EnCodec slices. <strong>UMAP</strong> uses the pooled corpus embedding; <strong>PCA-3D</strong> is fit on that file only (enable with <code>--intrinsic-pca</code> when building).</div>
      <label for="trajSound" class="sr-only">Sound for trajectory</label>
      <select id="trajSound" aria-label="Sound for trajectory polyline"></select>
      <div id="trajProgressRow" style="display:none">
        <span class="traj-progress-label" id="trajProgressLabel">Progress along path (slice)</span>
        <div class="traj-progress-row">
          <input type="range" id="trajProgress" min="0" max="0" step="1" value="0" aria-labelledby="trajProgressLabel" />
          <span id="trajProgressVal">—</span>
        </div>
      </div>
    </div>
    <div class="muted">
      Click a <strong>point</strong> or <strong>graph sphere</strong> for metadata. Hull meshes: click shows folder only.
      <span id="figureNote" style="display:block;margin-top:8px;color:#475569;"></span>
    </div>
  </aside>
  <div id="infoPanel">
    <h3>Selection</h3>
    <p class="muted" id="infoEmpty">Nothing selected. Per slice: 3D UMAP position (global corpus embedding), RGB (sound palette), folder id, sound id. Per sound: centroid, mean colour, slice count, label <code>folder/stem</code>. <strong>Trajectory modes</strong> connect slices in time order—UMAP is extrinsic; intrinsic PCA is per-file only. Audio decode is not available in this viewer.</p>
    <div id="infoBody" style="display:none"></div>
  </div>
  <div id="hint">Choose a <strong>View</strong>. Filter folders for points (left). Optional <strong>reference sound</strong> tints kNN edges by proximity. Drag orbit · scroll zoom · right-drag pan. HTTP only.</div>
  <script>
  window.addEventListener('error', function (e) {
    var d = document.createElement('pre');
    d.style.cssText = 'position:fixed;inset:0;padding:16px;background:#fff;color:#b91c1c;z-index:99999;overflow:auto;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap';
    d.textContent = (e && e.message) ? e.message : String(e);
    if (e && e.error && e.error.stack) d.textContent += '\n\n' + e.error.stack;
    document.body.appendChild(d);
  });
  window.addEventListener('unhandledrejection', function (e) {
    var r = e && e.reason;
    var d = document.createElement('pre');
    d.style.cssText = 'position:fixed;inset:0;padding:16px;background:#fff;color:#b91c1c;z-index:99999;overflow:auto;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap';
    d.textContent = 'Unhandled rejection: ' + (r && r.message ? r.message : String(r));
    if (r && r.stack) d.textContent += '\n\n' + r.stack;
    document.body.appendChild(d);
  });
  </script>
  <script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

  const META = __META_JSON__;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function decodeF32(b64, nFloats) {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Float32Array(raw.buffer, raw.byteOffset, nFloats);
  }
  function decodeU32(b64, nUints) {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Uint32Array(raw.buffer, raw.byteOffset, nUints);
  }

  function fillLegend() {
    const body = document.getElementById('legendBody');
    let html = '<div class="legend-section">Folders (hull colour)</div>';
    for (const row of META.legend.folders) {
      html += '<div class="legend-row"><span class="legend-swatch" style="background:' + escapeHtml(row.color) + '"></span><span>' + escapeHtml(row.name) + '</span><span class="legend-meta">' + row.count + '</span></div>';
    }
    html += '<div class="legend-section">Sounds (point / graph node colour)</div>';
    for (const row of META.legend.sounds) {
      const c = 'rgb(' + Math.round(row.r * 255) + ',' + Math.round(row.g * 255) + ',' + Math.round(row.b * 255) + ')';
      html += '<div class="legend-row"><span class="legend-swatch" style="background:' + c + '"></span><span>' + escapeHtml(row.label) + '</span><span class="legend-meta">' + row.n_slices + '</span></div>';
    }
    body.innerHTML = html;
  }
  fillLegend();

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.02, 200);
  camera.position.set(3.2, 2.4, 3.2);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const groupHull = new THREE.Group();
  groupHull.name = 'hulls';
  for (const H of META.hulls) {
    const pos = decodeF32(H.pos_b64, H.n_vertices * 3);
    const idx = decodeU32(H.idx_b64, H.n_triangles * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    // Raw TypedArray breaks indexed drawing in r160; must be a BufferAttribute.
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    // Solid-ish materials read on light backgrounds; heavy transmission looked "blank".
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(H.color),
      metalness: 0.22,
      roughness: 0.42,
      transparent: true,
      opacity: 0.82,
      side: THREE.DoubleSide,
      envMapIntensity: 0.95,
      depthWrite: true,
    });
    const mesh = new THREE.Mesh(g, mat);
    mesh.userData = { pickType: 'hull', folderName: H.name };
    mesh.renderOrder = 1;
    groupHull.add(mesh);
  }
  scene.add(groupHull);

  const groupGraph = new THREE.Group();
  groupGraph.name = 'graph';
  groupGraph.visible = false;
  const G = META.graph;
  const n = G.n_nodes;
  const nPos = decodeF32(G.nodes_pos_b64, n * 3);
  const nRgb = decodeF32(G.nodes_rgb_b64, n * 3);

  const sphere = new THREE.SphereGeometry(0.055, 20, 16);
  const im = new THREE.InstancedMesh(
    sphere,
    new THREE.MeshStandardMaterial({
      metalness: 0.12,
      roughness: 0.38,
      vertexColors: true,
      envMapIntensity: 0.9,
    }),
    n,
  );
  const dummy = new THREE.Object3D();
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    dummy.position.set(nPos[i * 3], nPos[i * 3 + 1], nPos[i * 3 + 2]);
    dummy.scale.setScalar(1);
    dummy.updateMatrix();
    im.setMatrixAt(i, dummy.matrix);
    c.setRGB(nRgb[i * 3], nRgb[i * 3 + 1], nRgb[i * 3 + 2], THREE.SRGBColorSpace);
    im.setColorAt(i, c);
  }
  im.instanceMatrix.needsUpdate = true;
  if (im.instanceColor) im.instanceColor.needsUpdate = true;
  im.renderOrder = 2;
  groupGraph.add(im);

  const linePos = [];
  for (const [a, b] of G.edges) {
    linePos.push(
      nPos[a * 3], nPos[a * 3 + 1], nPos[a * 3 + 2],
      nPos[b * 3], nPos[b * 3 + 1], nPos[b * 3 + 2],
    );
  }
  const nEdgeVerts = linePos.length;
  const edgeColors = new Float32Array(nEdgeVerts);
  for (let i = 0; i < nEdgeVerts; i += 3) {
    edgeColors[i] = 0.58;
    edgeColors[i + 1] = 0.62;
    edgeColors[i + 2] = 0.70;
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
  lg.setAttribute('color', new THREE.Float32BufferAttribute(edgeColors, 3));
  const lines = new THREE.LineSegments(
    lg,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
    }),
  );
  lines.renderOrder = 0;
  groupGraph.add(lines);
  scene.add(groupGraph);

  const groupPoints = new THREE.Group();
  groupPoints.name = 'points';
  groupPoints.visible = false;
  const Pt = META.points;
  const PSTR = Pt.stride || 8;
  const pbuf = decodeF32(Pt.interp_b64, Pt.count * PSTR);
  const fullPN = Pt.count;
  const fullPpos = new Float32Array(fullPN * 3);
  const fullPcol = new Float32Array(fullPN * 3);
  const fullPfolder = new Int32Array(fullPN);
  const fullPsound = new Int32Array(fullPN);
  for (let i = 0; i < fullPN; i++) {
    const o = i * PSTR;
    fullPpos[i * 3] = pbuf[o];
    fullPpos[i * 3 + 1] = pbuf[o + 1];
    fullPpos[i * 3 + 2] = pbuf[o + 2];
    fullPcol[i * 3] = pbuf[o + 3];
    fullPcol[i * 3 + 1] = pbuf[o + 4];
    fullPcol[i * 3 + 2] = pbuf[o + 5];
    fullPfolder[i] = Math.round(pbuf[o + 6]);
    fullPsound[i] = Math.round(pbuf[o + 7]);
  }

  const pgeom = new THREE.BufferGeometry();
  let pointParentIndex = new Int32Array(0);
  const pmat = new THREE.PointsMaterial({
    size: 0.038,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    sizeAttenuation: true,
    depthWrite: true,
  });
  const pointsObj = new THREE.Points(pgeom, pmat);
  pointsObj.renderOrder = 3;
  groupPoints.add(pointsObj);
  scene.add(groupPoints);

  const groupTraj = new THREE.Group();
  groupTraj.name = 'trajectory';
  groupTraj.visible = false;
  scene.add(groupTraj);

  const figureNoteEl = document.getElementById('figureNote');
  const NOTES = META.figure_notes || {};
  const trajProgressRow = document.getElementById('trajProgressRow');
  const trajProgress = document.getElementById('trajProgress');
  const trajProgressVal = document.getElementById('trajProgressVal');

  let trajPlayhead = null;
  let trajPosBuf = null;
  let trajSliceN = 0;

  function resetTrajProgressState() {
    trajPlayhead = null;
    trajPosBuf = null;
    trajSliceN = 0;
    if (trajProgressRow) trajProgressRow.style.display = 'none';
    if (trajProgressVal) trajProgressVal.textContent = '—';
  }

  function applyTrajProgressIndex(i) {
    if (!trajPosBuf || !trajPlayhead || trajSliceN <= 0) return;
    const idx = Math.max(0, Math.min(trajSliceN - 1, i | 0));
    const o = idx * 3;
    trajPlayhead.position.set(trajPosBuf[o], trajPosBuf[o + 1], trajPosBuf[o + 2]);
    if (trajProgressVal) trajProgressVal.textContent = trajSliceN <= 0 ? '—' : `${idx + 1} / ${trajSliceN}`;
  }

  function disposeTrajectoryChildren() {
    while (groupTraj.children.length > 0) {
      const ch = groupTraj.children[groupTraj.children.length - 1];
      groupTraj.remove(ch);
      if (ch.geometry) ch.geometry.dispose();
      if (ch.material) ch.material.dispose();
    }
  }

  function rebuildTrajectoryLine(space) {
    disposeTrajectoryChildren();
    resetTrajProgressState();
    const trajSel = document.getElementById('trajSound');
    const sid = parseInt(trajSel.value, 10);
    if (!Number.isFinite(sid) || sid < 0) {
      figureNoteEl.textContent = 'Pick a sound for the trajectory.';
      return;
    }
    const list = space === 'intrinsic' ? META.trajectories_intrinsic : META.trajectories_umap;
    const entry = list && list[sid];
    if (!entry || !entry.pos_b64) {
      figureNoteEl.textContent = space === 'intrinsic'
        ? 'No intrinsic PCA row for this id — rebuild viewer with --intrinsic-pca and ensure .npy exists under data/enc24k.'
        : 'No UMAP trajectory (file has fewer than 2 slices in export).';
      return;
    }
    figureNoteEl.textContent = space === 'intrinsic' ? (NOTES.intrinsic_pca || '') : (NOTES.global_umap || '');
    const pos = decodeF32(entry.pos_b64, entry.n * 3);
    const col = new Float32Array(entry.n * 3);
    for (let i = 0; i < entry.n; i++) {
      const u = entry.n > 1 ? i / (entry.n - 1) : 0;
      col[i * 3] = 0.15 + 0.85 * u;
      col[i * 3 + 1] = 0.15 + 0.75 * Math.sin(u * Math.PI);
      col[i * 3 + 2] = 0.75 * (1 - u) + 0.1;
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    lg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const line = new THREE.Line(
      lg,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.98 }),
    );
    line.renderOrder = 4;
    groupTraj.add(line);

    trajSliceN = entry.n;
    trajPosBuf = pos;
    const phGeom = new THREE.SphereGeometry(0.042, 22, 16);
    const phMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x2563eb,
      emissiveIntensity: 0.55,
      metalness: 0.15,
      roughness: 0.35,
    });
    trajPlayhead = new THREE.Mesh(phGeom, phMat);
    trajPlayhead.renderOrder = 5;
    groupTraj.add(trajPlayhead);

    if (trajProgressRow) trajProgressRow.style.display = '';
    if (trajProgress) {
      trajProgress.min = '0';
      trajProgress.max = String(Math.max(0, entry.n - 1));
      trajProgress.step = '1';
      trajProgress.value = '0';
    }
    applyTrajProgressIndex(0);
  }

  function distCentroids(ai, bi) {
    const ax = nPos[ai * 3] - nPos[bi * 3];
    const ay = nPos[ai * 3 + 1] - nPos[bi * 3 + 1];
    const az = nPos[ai * 3 + 2] - nPos[bi * 3 + 2];
    return Math.sqrt(ax * ax + ay * ay + az * az);
  }

  function distSliceToCentroid(parentIdx, nodeIdx) {
    const ax = fullPpos[parentIdx * 3] - nPos[nodeIdx * 3];
    const ay = fullPpos[parentIdx * 3 + 1] - nPos[nodeIdx * 3 + 1];
    const az = fullPpos[parentIdx * 3 + 2] - nPos[nodeIdx * 3 + 2];
    return Math.sqrt(ax * ax + ay * ay + az * az);
  }

  function updateEdgeColors(refVal) {
    const colAttr = lines.geometry.getAttribute('color');
    const greyAll = () => {
      for (let i = 0; i < colAttr.count; i++) {
        colAttr.setXYZ(i, 0.58, 0.62, 0.70);
      }
      colAttr.needsUpdate = true;
    };
    if (refVal === '' || refVal === null || refVal === undefined) {
      greyAll();
      return;
    }
    const rid = parseInt(refVal, 10);
    if (!Number.isFinite(rid) || rid < 0 || rid >= n) {
      greyAll();
      return;
    }
    const dEach = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      dEach[i] = distCentroids(i, rid);
    }
    let dmin = Infinity;
    let dmax = -Infinity;
    for (const [a, b] of G.edges) {
      const t = Math.min(dEach[a], dEach[b]);
      if (t < dmin) dmin = t;
      if (t > dmax) dmax = t;
    }
    const span = Math.max(dmax - dmin, 1e-8);
    let ei = 0;
    for (const [a, b] of G.edges) {
      const t = Math.min(dEach[a], dEach[b]);
      const u = 1 - (t - dmin) / span;
      const r = 0.12 + 0.88 * u;
      const g = 0.22 + 0.5 * (1 - u);
      const bVal = 0.08 + 0.72 * (1 - u);
      colAttr.setXYZ(ei * 2, r, g, bVal);
      colAttr.setXYZ(ei * 2 + 1, r, g, bVal);
      ei++;
    }
    colAttr.needsUpdate = true;
  }

  function rebuildPointsFilter() {
    const allowed = new Set();
    document.querySelectorAll('.folder-filter-cb').forEach((cb) => {
      if (cb.checked) allowed.add(parseInt(cb.dataset.fid, 10));
    });
    if (allowed.size === 0) {
      pgeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      pgeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
      pointParentIndex = new Int32Array(0);
      pgeom.computeBoundingSphere();
      return;
    }
    let outN = 0;
    for (let i = 0; i < fullPN; i++) {
      if (allowed.has(fullPfolder[i])) outN++;
    }
    const outP = new Float32Array(outN * 3);
    const outC = new Float32Array(outN * 3);
    pointParentIndex = new Int32Array(outN);
    let w = 0;
    for (let i = 0; i < fullPN; i++) {
      if (!allowed.has(fullPfolder[i])) continue;
      outP[w * 3] = fullPpos[i * 3];
      outP[w * 3 + 1] = fullPpos[i * 3 + 1];
      outP[w * 3 + 2] = fullPpos[i * 3 + 2];
      outC[w * 3] = fullPcol[i * 3];
      outC[w * 3 + 1] = fullPcol[i * 3 + 1];
      outC[w * 3 + 2] = fullPcol[i * 3 + 2];
      pointParentIndex[w] = i;
      w++;
    }
    pgeom.setAttribute('position', new THREE.BufferAttribute(outP, 3));
    pgeom.setAttribute('color', new THREE.BufferAttribute(outC, 3));
    pgeom.computeBoundingSphere();
  }

  const mount = document.getElementById('folderFilterMount');
  META.folder_names.forEach((fn, fid) => {
    const lab = document.createElement('label');
    lab.className = 'cb';
    const inp = document.createElement('input');
    inp.type = 'checkbox';
    inp.className = 'folder-filter-cb';
    inp.id = 'fcb-' + fid;
    inp.dataset.fid = String(fid);
    inp.checked = true;
    inp.addEventListener('change', rebuildPointsFilter);
    lab.appendChild(inp);
    lab.appendChild(document.createTextNode(' ' + fn));
    mount.appendChild(lab);
  });

  const refSel = document.getElementById('refSound');
  META.folder_names.forEach((fn) => {
    const subs = META.sound_meta.filter((s) => s.folder === fn);
    if (!subs.length) return;
    const og = document.createElement('optgroup');
    og.label = fn;
    subs.forEach((s) => {
      const o = document.createElement('option');
      o.value = String(s.id);
      o.textContent = s.stem;
      og.appendChild(o);
    });
    refSel.appendChild(og);
  });
  refSel.addEventListener('change', () => updateEdgeColors(refSel.value));

  const trajSound = document.getElementById('trajSound');
  META.folder_names.forEach((fn) => {
    const subs = META.sound_meta.filter((s) => s.folder === fn);
    if (!subs.length) return;
    const og = document.createElement('optgroup');
    og.label = fn;
    subs.forEach((s) => {
      const o = document.createElement('option');
      o.value = String(s.id);
      o.textContent = s.stem;
      og.appendChild(o);
    });
    trajSound.appendChild(og);
  });
  trajSound.addEventListener('change', () => {
    const v = document.getElementById('viewSelect').value;
    if (v === 'traj_umap') rebuildTrajectoryLine('umap');
    else if (v === 'traj_intrinsic') rebuildTrajectoryLine('intrinsic');
  });
  if (trajProgress) {
    trajProgress.addEventListener('input', () => {
      applyTrajProgressIndex(parseInt(trajProgress.value, 10));
    });
  }

  rebuildPointsFilter();
  updateEdgeColors('');

  const infoEmpty = document.getElementById('infoEmpty');
  const infoBody = document.getElementById('infoBody');
  function showInfo(html) {
    if (!html) {
      infoEmpty.style.display = '';
      infoBody.style.display = 'none';
      infoBody.innerHTML = '';
      return;
    }
    infoEmpty.style.display = 'none';
    infoBody.style.display = '';
    infoBody.innerHTML = html;
  }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.075 };
  const pointer = new THREE.Vector2();

  function fmtDist(d) {
    if (!Number.isFinite(d)) return '—';
    return d.toFixed(4);
  }

  renderer.domElement.addEventListener('click', (ev) => {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const objs = [];
    if (groupPoints.visible) objs.push(pointsObj);
    if (groupGraph.visible) objs.push(im);
    if (groupHull.visible) {
      groupHull.children.forEach((ch) => objs.push(ch));
    }
    const hits = raycaster.intersectObjects(objs, false);
    if (!hits.length) {
      showInfo('');
      return;
    }
    const h = hits[0];
    if (h.object === pointsObj && h.index !== undefined && pointParentIndex.length) {
      const parentIdx = pointParentIndex[h.index];
      const sid = fullPsound[parentIdx];
      const sm = META.sound_meta[sid];
      const refVal = refSel.value;
      let dRefHtml = '';
      if (refVal !== '' && Number.isFinite(parseInt(refVal, 10))) {
        const R = parseInt(refVal, 10);
        dRefHtml = '<dt>Distance to reference centroid</dt><dd>' + fmtDist(distSliceToCentroid(parentIdx, R)) + ' (Euclidean in normalized 3D UMAP)</dd>';
      }
      const px = fullPpos[parentIdx * 3];
      const py = fullPpos[parentIdx * 3 + 1];
      const pz = fullPpos[parentIdx * 3 + 2];
      showInfo(
        '<h3>Embedding slice (point)</h3><dl>'
        + '<dt>What this is</dt><dd>One row of a slice embedding; colour = sound palette.</dd>'
        + '<dt>Sound</dt><dd>' + escapeHtml(sm.label) + '</dd>'
        + '<dt>Folder</dt><dd>' + escapeHtml(sm.folder) + '</dd>'
        + '<dt>Stem</dt><dd>' + escapeHtml(sm.stem) + '</dd>'
        + '<dt>Slices for this sound (full data)</dt><dd>' + sm.n_slices + '</dd>'
        + '<dt>UMAP xyz (normalized)</dt><dd>' + px.toFixed(4) + ', ' + py.toFixed(4) + ', ' + pz.toFixed(4) + '</dd>'
        + dRefHtml
        + '</dl>',
      );
      return;
    }
    if (h.object === im && h.instanceId !== undefined) {
      const sid = h.instanceId;
      const sm = META.sound_meta[sid];
      const rx = nPos[sid * 3];
      const ry = nPos[sid * 3 + 1];
      const rz = nPos[sid * 3 + 2];
      const refVal = refSel.value;
      let dRefHtml = '';
      if (refVal !== '' && Number.isFinite(parseInt(refVal, 10))) {
        const R = parseInt(refVal, 10);
        dRefHtml = '<dt>Centroid distance to reference</dt><dd>' + fmtDist(distCentroids(sid, R)) + '</dd>';
      }
      showInfo(
        '<h3>Sound (graph node)</h3><dl>'
        + '<dt>What this is</dt><dd>Centroid of all slices of this file in 3D UMAP.</dd>'
        + '<dt>Label</dt><dd>' + escapeHtml(sm.label) + '</dd>'
        + '<dt>Folder</dt><dd>' + escapeHtml(sm.folder) + '</dd>'
        + '<dt>Stem</dt><dd>' + escapeHtml(sm.stem) + '</dd>'
        + '<dt>Slice count</dt><dd>' + sm.n_slices + '</dd>'
        + '<dt>Centroid xyz (normalized)</dt><dd>' + rx.toFixed(4) + ', ' + ry.toFixed(4) + ', ' + rz.toFixed(4) + '</dd>'
        + dRefHtml
        + '</dl>',
      );
      return;
    }
    if (h.object.userData && h.object.userData.pickType === 'hull') {
      showInfo(
        '<h3>Folder hull</h3><dl>'
        + '<dt>Folder</dt><dd>' + escapeHtml(h.object.userData.folderName) + '</dd>'
        + '<dt>What this is</dt><dd>Convex hull of subsampled UMAP points for this folder.</dd></dl>',
      );
    }
  });

  scene.add(new THREE.AmbientLight(0xffffff, 0.42));
  const dl = new THREE.DirectionalLight(0xffffff, 0.75);
  dl.position.set(4, 6, 3);
  scene.add(dl);

  const viewSelect = document.getElementById('viewSelect');
  const hint = document.getElementById('hint');

  function setMode(which) {
    const isTrajU = which === 'traj_umap';
    const isTrajI = which === 'traj_intrinsic';
    groupHull.visible = which === 'hull';
    groupGraph.visible = which === 'graph';
    groupPoints.visible = which === 'points';
    groupTraj.visible = isTrajU || isTrajI;
    viewSelect.value = which;
    figureNoteEl.textContent = '';
    if (!isTrajU && !isTrajI && trajProgressRow) trajProgressRow.style.display = 'none';
    if (isTrajU) {
      rebuildTrajectoryLine('umap');
      hint.innerHTML = '<strong>Sound trajectory (UMAP)</strong> — time-ordered path in the pooled corpus embedding (extrinsic). Use <strong>Progress along path</strong> to scrub the slice loop. ' + (NOTES.decode ? escapeHtml(NOTES.decode) : '');
    } else if (isTrajI) {
      rebuildTrajectoryLine('intrinsic');
      hint.innerHTML = '<strong>Intrinsic PCA-3D</strong> — fitted on one file’s (T,128) slice matrix only. Use <strong>Progress along path</strong> to scrub slices in time order. ' + (NOTES.decode ? escapeHtml(NOTES.decode) : '');
    } else if (which === 'hull') {
      hint.innerHTML = '<strong>Folder hulls</strong> — convex shells (subsampled slices per folder). Open <strong>Legend</strong> for folder / sound keys.';
    } else if (which === 'graph') {
      hint.innerHTML = '<strong>File graph</strong> — kNN edges; pick a <strong>reference sound</strong> to heat edges by centroid proximity. Click spheres for metadata.';
    } else {
      hint.innerHTML = '<strong>Points</strong> — subsampled slices; use folder checkboxes to show/hide. Click a point for fields + optional distance to reference.';
    }
  }

  viewSelect.addEventListener('change', () => setMode(viewSelect.value));

  if (META.hulls.length === 0 && Pt.count > 0) {
    setMode('points');
    hint.innerHTML = '<strong>No hull geometry</strong>. Showing <strong>Points</strong> instead. Try <strong>File graph</strong> from the View menu.';
  } else if (META.hulls.length === 0 && G.n_nodes > 0) {
    setMode('graph');
    hint.innerHTML = '<strong>No hull geometry</strong>. Showing <strong>File graph</strong> instead.';
  } else {
    setMode('hull');
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();
  </script>
</body>
</html>
"""

    out = html.replace("__META_JSON__", json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Wrote {args.out}")
    print(
        f"  Hulls: {len(hulls)} folders, graph: {n_sounds} nodes / {len(edges)} edges, "
        f"points: {meta['points']['count']}"
    )


if __name__ == "__main__":
    main()
