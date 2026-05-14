#!/usr/bin/env python3
"""
Build a self-contained interactive HTML viewer for 3D UMAP with a liquid-glass
look (Three.js: MeshPhysicalMaterial + transmission + RoomEnvironment).

Each enc24k folder (sound category) uses a different glass recipe: transmission,
thickness, roughness, IOR, and volume attenuation vary by type.

Usage:
  python3 scripts/build_umap_glass_viewer.py
  python3 scripts/build_umap_glass_viewer.py --max-points 20000

Open the generated HTML via a local server (required for ES modules), e.g.:
  cd results/PCA_plots && python3 -m http.server 8765
  → http://localhost:8765/enc24k_umap_glass_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Any

import numpy as np


# Per-folder glass tuning (keys must match directory names under data/enc24k).
# Higher transmission → clearer; higher roughness → frostier; thickness/ior shape refraction.
GLASS_BY_FOLDER: dict[str, dict[str, float]] = {
    "breaks": {
        "transmission": 0.82,
        "thickness": 0.82,
        "roughness": 0.28,
        "ior": 1.38,
        "attenuation_distance": 0.28,
        "env_map_intensity": 0.88,
        "clearcoat": 0.35,
        "clearcoat_roughness": 0.4,
    },
    "instrument_samples": {
        "transmission": 0.96,
        "thickness": 0.38,
        "roughness": 0.07,
        "ior": 1.52,
        "attenuation_distance": 0.62,
        "env_map_intensity": 1.22,
        "clearcoat": 0.85,
        "clearcoat_roughness": 0.06,
    },
    "kalimbas": {
        "transmission": 0.9,
        "thickness": 0.55,
        "roughness": 0.14,
        "ior": 1.46,
        "attenuation_distance": 0.45,
        "env_map_intensity": 1.05,
        "clearcoat": 0.5,
        "clearcoat_roughness": 0.18,
    },
    "music": {
        "transmission": 0.98,
        "thickness": 0.32,
        "roughness": 0.05,
        "ior": 1.55,
        "attenuation_distance": 0.75,
        "env_map_intensity": 1.28,
        "clearcoat": 0.92,
        "clearcoat_roughness": 0.04,
    },
    "sound_effects": {
        "transmission": 0.78,
        "thickness": 0.68,
        "roughness": 0.2,
        "ior": 1.44,
        "attenuation_distance": 0.35,
        "env_map_intensity": 1.0,
        "clearcoat": 0.45,
        "clearcoat_roughness": 0.28,
    },
    "soundscapes": {
        "transmission": 0.92,
        "thickness": 0.72,
        "roughness": 0.12,
        "ior": 1.48,
        "attenuation_distance": 0.4,
        "env_map_intensity": 1.08,
        "clearcoat": 0.25,
        "clearcoat_roughness": 0.35,
    },
    "speech": {
        "transmission": 0.88,
        "thickness": 0.48,
        "roughness": 0.11,
        "ior": 1.5,
        "attenuation_distance": 0.55,
        "env_map_intensity": 1.12,
        "clearcoat": 0.65,
        "clearcoat_roughness": 0.1,
    },
}

_DEFAULT_GLASS: dict[str, float] = {
    "transmission": 0.9,
    "thickness": 0.5,
    "roughness": 0.12,
    "ior": 1.48,
    "attenuation_distance": 0.5,
    "env_map_intensity": 1.1,
    "clearcoat": 0.5,
    "clearcoat_roughness": 0.15,
}


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = 0.5 * (lo + hi)
    span = np.max(hi - lo)
    if span <= 1e-12:
        span = 1.0
    return (xyz - c) * (2.2 / span)


def _presets_for_folders(folder_names: list[str]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for name in folder_names:
        g = GLASS_BY_FOLDER.get(str(name), _DEFAULT_GLASS)
        out.append(dict(g))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        default="results/PCA_plots/enc24k_umap_3d_export.npz",
        help="Output from pca_enc24k_all.py (UMAP block)",
    )
    p.add_argument(
        "--out",
        default="results/PCA_plots/enc24k_umap_glass_viewer.html",
        help="Written self-contained HTML",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=25_000,
        help="Cap point count for smooth glass instancing (random subsample)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.npz)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    fid = np.asarray(data["folder_id"], dtype=np.int32)
    n = xyz.shape[0]

    folder_names = [str(x) for x in np.asarray(data["folders"]).tolist()]
    glass_presets: list[dict[str, Any]] = _presets_for_folders(folder_names)

    rng = np.random.default_rng(args.seed)
    if n > args.max_points:
        idx = rng.choice(n, size=args.max_points, replace=False)
        xyz, rgb, fid = xyz[idx], rgb[idx], fid[idx]
        n = args.max_points

    xyz = _normalize_xyz(xyz).astype(np.float32)
    # xyz(3) + rgb(3) + folder_id(1) as f32 for a single blob
    inter = np.empty((n, 7), dtype=np.float32)
    inter[:, :3] = xyz
    inter[:, 3:6] = np.clip(rgb, 0.0, 1.0)
    inter[:, 6] = fid.astype(np.float32)
    blob_b64 = base64.b64encode(inter.tobytes()).decode("ascii")

    meta = {
        "count": int(n),
        "blob_b64": blob_b64,
        "layout": "interleaved_f32_xyzrgb_folder_rowmajor",
        "folder_names": folder_names,
        "glass_presets": glass_presets,
    }

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UMAP 3D — liquid glass by sound type</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #ffffff; }
    #hint {
      position: fixed; left: 12px; bottom: 10px; z-index: 10;
      font: 13px/1.35 system-ui, sans-serif; color: #334155; max-width: 480px;
      background: rgba(255,255,255,0.72); backdrop-filter: blur(10px);
      border: 1px solid rgba(148,163,184,0.35); border-radius: 10px; padding: 10px 12px;
    }
    #hint strong { color: #0f172a; }
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
  <div id="hint"><strong>Glass varies by folder</strong> — clearer vs frosted vs thick “liquid” is driven by transmission, roughness, thickness, and IOR per sound category. Drag orbit · scroll zoom · right-drag pan. Use HTTP server (not file://).</div>
  <script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

  const META = __META_JSON__;
  const STRIDE = 7;

  function decodeInterleaved(blobB64, count) {
    const raw = Uint8Array.from(atob(blobB64), c => c.charCodeAt(0));
    const f = new Float32Array(raw.buffer, raw.byteOffset, count * STRIDE);
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const folder = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE;
      pos[i * 3] = f[o]; pos[i * 3 + 1] = f[o + 1]; pos[i * 3 + 2] = f[o + 2];
      col[i * 3] = f[o + 3]; col[i * 3 + 1] = f[o + 4]; col[i * 3 + 2] = f[o + 5];
      folder[i] = Math.round(f[o + 6]);
    }
    return { pos, col, folder };
  }

  function bucketPoints(pos, col, folder, nFolders) {
    const counts = new Array(nFolders).fill(0);
    for (let i = 0; i < folder.length; i++) {
      const fi = folder[i];
      if (fi >= 0 && fi < nFolders) counts[fi]++;
    }
    const posOut = counts.map((c) => new Float32Array(c * 3));
    const colOut = counts.map((c) => new Float32Array(c * 3));
    const ptr = new Array(nFolders).fill(0);
    for (let i = 0; i < folder.length; i++) {
      const fi = folder[i];
      if (fi < 0 || fi >= nFolders) continue;
      const p = ptr[fi]++;
      const o = p * 3;
      posOut[fi][o] = pos[i * 3]; posOut[fi][o + 1] = pos[i * 3 + 1]; posOut[fi][o + 2] = pos[i * 3 + 2];
      colOut[fi][o] = col[i * 3]; colOut[fi][o + 1] = col[i * 3 + 1]; colOut[fi][o + 2] = col[i * 3 + 2];
    }
    return { posOut, colOut, counts };
  }

  function makeGlassMaterial(preset) {
    return new THREE.MeshPhysicalMaterial({
      transmission: preset.transmission,
      thickness: preset.thickness,
      roughness: preset.roughness,
      metalness: 0.0,
      ior: preset.ior,
      transparent: true,
      envMapIntensity: preset.env_map_intensity,
      attenuationColor: new THREE.Color(0xffffff),
      attenuationDistance: preset.attenuation_distance,
      clearcoat: preset.clearcoat,
      clearcoatRoughness: preset.clearcoat_roughness,
    });
  }

  const count = META.count;
  const { pos, col, folder } = decodeInterleaved(META.blob_b64, count);
  const nFolders = META.folder_names.length;
  const { posOut, colOut, counts } = bucketPoints(pos, col, folder, nFolders);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
  camera.position.set(2.8, 2.2, 2.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const geo = new THREE.SphereGeometry(0.028, 14, 10);
  const dummy = new THREE.Object3D();
  const c = new THREE.Color();

  for (let fi = 0; fi < nFolders; fi++) {
    const n = counts[fi];
    if (n === 0) continue;
    const preset = META.glass_presets[fi];
    const mat = makeGlassMaterial(preset);
    const mesh = new THREE.InstancedMesh(geo, mat, n);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const pp = posOut[fi];
    const cc = colOut[fi];
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      dummy.position.set(pp[o], pp[o + 1], pp[o + 2]);
      const lum = cc[o] * 0.3 + cc[o + 1] * 0.59 + cc[o + 2] * 0.11;
      const s = 0.9 + 0.1 * lum;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      c.setRGB(cc[o], cc[o + 1], cc[o + 2], THREE.SRGBColorSpace);
      mesh.setColorAt(i, c);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.name = META.folder_names[fi];
    scene.add(mesh);
  }

  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(3, 5, 4);
  scene.add(dir);

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

    meta_json = json.dumps(meta)
    out = html.replace("__META_JSON__", meta_json)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {args.out} ({n} points, {len(folder_names)} folder glass presets).")


if __name__ == "__main__":
    main()
