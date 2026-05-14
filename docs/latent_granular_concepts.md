# Latent granular concepts (EnCodec embeddings)

This note defines the terms used when we talk about **latent granular** analysis and the **interactive 3D viewers** in this project. It matches how the scripts use your data: frozen **EnCodec** rows per clip, stored as `.npy` arrays under `data/enc24k/<category>/`.

---

## Slice embedding (frame)

Each **row** of a `(T, 128)` array is one **slice** (or **frame**) of latents for that audio file: a point in \(\mathbb{R}^{128}\) at one time step along the encoder’s temporal grid. Rows are in **time order** for a single file.

- **\(T\)**: number of slices in that file (depends on clip length and encoder hop).
- **128**: EnCodec latent dimension for this export (fixed for the dataset).

---

## Grain

A **grain** is a **small chunk of consecutive slices** that we treat as one feature vector. We build grains by sliding a window along time:

| Parameter | Meaning |
|-----------|---------|
| **Window size** | How many consecutive rows are grouped into one grain. |
| **Stride** | Step between window starts (often equal to window size for non-overlapping blocks). |
| **Grain mode** | **`mean`**: average the rows in the window → vector remains 128-D. **`concat`**: stack the rows into one long vector (dimension \(128 \times \text{window size}\)). |

**Why grains exist:** A single slice can be very local; a grain matches the idea of a short **acoustic token** used when matching one sound to a **codebook** built from another.

---

## Codebook

The **codebook** is the set of **candidate grain vectors** taken from one or more **source** recordings (usually everything you load from a **source folder**, e.g. `instrument_samples/`). After normalization (typically **L2 per row**), it is a matrix whose rows are “donor” patterns you can match **against**.

- **Size**: \(N_{\text{cb}} \times D\) with \(D = 128\) for `mean` grains (or larger for `concat`).
- **Role**: In **latent granular resynthesis** (see e.g. *Latent Granular Resynthesis using Neural Audio Codecs*, ISMIR 2025), the codebook is where you **borrow** latents from when building a hybrid representation of the **target**.

In plots and the 3D viewer, the codebook often appears as a **cloud of points** (subsampled if there are many grains).

---

## Target

The **target** is the audio (as its EnCodec `.npy`) you want to **analyze or resynthesize**. We turn it into a sequence of **target grains** with the **same** window size, stride, and grain mode as for the codebook, so each target grain is comparable in dimension to codebook rows.

- **Size**: \(M \times D\) after grain extraction (with \(M\) target time steps at the grain level).
- **Role**: For each target time index \(t\), we ask: “**which codebook row (or distribution over rows)** best matches this target grain?”

---

## Matching (cosine and softmax)

**Similarity** between a target grain \(z_t\) and every codebook row is often measured as **cosine similarity** after L2 normalization (equivalent to a dot product of unit vectors).

| Concept | Meaning |
|---------|---------|
| **Argmax match** | Pick the codebook index with **highest** cosine similarity to \(z_t\). |
| **Margin** | Difference between the best and second-best cosine scores at step \(t\). Large margin ⇒ a **clear** winner; small margin ⇒ **ambiguous** match. |
| **Softmax over codebook** | Turn similarities into a **probability vector** over codebook indices using temperature **τ**. Smaller τ sharpens the distribution (closer to argmax); larger τ flattens it. |
| **Entropy** | Shannon entropy of that distribution. **High** entropy ⇒ spread mass (many plausible donors); **low** ⇒ peaked (confident pick). |

These quantities are diagnostic: they describe **how stable** the mapping from target time to codebook is, not whether it sounds good (that needs **decode**).

---

## Trajectories

A **trajectory** is an ordered polyline in latent space:

- **Target trajectory**: connect target grains \(z_0, z_1, \ldots\) in **time order**. Shows how the clip **moves** in representation space as it plays.
- **Argmax walk**: connect the **codebook vectors** chosen at each \(t\) by argmax. This path can **jump** sharply when the best donor switches—unlike a smooth path in the outer world.

Trajectories can be shown in different **3D charts** (not the same object—read captions on each figure):

1. **Global UMAP** (`enc24k_umap_structures_viewer.html`): 3D UMAP trained on **all** pooled slices from many files. **Extrinsic** view of “where this file lives” vs the rest of the corpus.
2. **Intrinsic PCA-3D (per file)** (optional in the same viewer): PCA fit on **one** file’s `(T,128)` only. **Intrinsic** summary without competition from other recordings.
3. **Joint PCA-3D** (`latent_granular_3d_viewer.html`): PCA fit on stacked **[codebook subsample ; target grains]**. Puts **donors and target** in one **shared** 3D map for **matching geometry** (good for orbiting codebook vs target path). The page has dropdowns for **codebook source folder** and **target sound**; each pair needs a small JSON file under `bundles/`. Precompute all `folder × target` pairs with `python3 scripts/build_latent_granular_3d_viewer.py --build-all-bundles --out …/latent_granular_3d_viewer.html` (optionally `--skip-existing` on reruns); the page can **prefetch** every bundle for the selected codebook source into memory for **near–real-time** target switching, or **cycle** through stems automatically.

   **Global PCA / global UMAP (same viewer):** Matching and the argmax walk are still computed in **L2-normalized grain space** (cosine / dot-product geometry). For **display**, you can optionally add corpus-wide 3D maps fit on **pooled grains** (same window, stride, and grain mode as the bundle). Run a one-time export:

   ```bash
   python3 scripts/export_global_grain_embedding.py --enc-root data/enc24k --window-size 1 --stride 1 --grain-mode mean --out-dir results/PCA_plots/
   ```

   Then build bundles with `--global-embed-npz …_pca.npz` and optionally `--global-umap-model …_umap.joblib` (the `.npz` fingerprint must match the bundle grain settings). The viewer shows an **Embedding space** control: **Joint PCA** (pairwise), **Global PCA**, or **Global UMAP**. **Tight hull** meshes (alpha shapes) are emitted for **joint PCA** only by default, to keep JSON size reasonable; global spaces reuse the same trajectories with different xyz only. For joint PCA hulls and codebook styling in the page, see **SciPy** at export time, `--hull-alpha-percentile`, `--no-tight-hulls`, and the **Codebook look** dropdown (points / envelope / solids).

4. **Spectral / eigenmap** (2D PNGs from `latent_granular_viz.py`): embedding of a **subsampled codebook** with graph-based affinity; the target is mapped via **nearest neighbors** in that subsample—useful but **hyperparameter-sensitive** (`k`, subsample size).

---

## Global grain embedding export (for the granular 3D viewer)

The script `export_global_grain_embedding.py` pools **L2-normalized grains** from all category folders under `--enc-root` (same scanning and grain rules as `build_latent_granular_3d_viewer.py`). It fits **PCA(3)** on the stacked grain matrix and **UMAP(3)** (cosine metric) on a subsample if needed, and writes:

- `enc24k_global_grain_{fingerprint}_pca.npz` — `mean`, `components`, **corpus-wide normalization** scalars (`corpus_norm_pca_*`, `corpus_norm_umap_*`) so bundle trajectories align with the background cloud, plus metadata (`fingerprint`, `d`, grain counts, etc.)
- `enc24k_global_grain_{fingerprint}_umap.joblib` — fitted UMAP for `.transform()` at bundle build time
- `enc24k_global_grain_{fingerprint}_corpus_cloud.json` — a subsampled **full corpus** in both `global_pca` and `global_umap` 3D (base64 positions + per-point folder id). Each **category folder** gets a distinct **colour** and **marker shape** so the latent granular viewer can draw the same cloud as context behind the target / argmax tubes. Tune size with `--max-corpus-cloud`.

The **fingerprint** (e.g. `w1_s1_mean`) encodes window size, stride, and grain mode so `pack_bundle_meta` can reject mismatched exports. This is **not** the same object as slice-level UMAP from `pca_enc24k_all.py`: bundles use **grains**, so the global map must be trained in grain feature space when windows overlap or concat mode is used.

The **fingerprint** (e.g. `w1_s1_mean`) encodes window size, stride, and grain mode so `pack_bundle_meta` can reject mismatched exports. This is **not** the same object as slice-level UMAP from `pca_enc24k_all.py`: bundles use **grains**, so the global map must be trained in grain feature space when windows overlap or concat mode is used.

| Build flag | Role |
|------------|------|
| `--global-embed-npz` | Path to the PCA `.npz` from the export (includes corpus normalization when re-exported) |
| `--global-umap-model` | Optional path to the `.joblib` UMAP model (requires the npz for validation) |

Building bundles with those flags copies the sibling `*_corpus_cloud.json` into `bundles/` and sets `global_corpus_cloud` on `manifest.json` so the HTML viewer can load the full background cloud in global embedding modes.

---

## Internal geometry (histogram method)

Separately from codebook–target matching, **internal geometry** (`compare_internal_geometry.py`) summarizes **within-file** structure: random pairs of grains from **one** file, Euclidean distances, histogram of those distances. Comparing histograms across files tells you whether **relational spread** inside clips is similar, **without** a global UMAP.

---

## What the scripts produce (quick map)

| Script | Main idea |
|--------|-----------|
| `enc24k_geometry_checks.py` | Is the time series smooth? Chord vs path length? Per-file PCA variance? |
| `latent_granular_viz.py` | Matching time series, heatmaps, spectral plot, NN walk, local PCA angles, optional softmax KL. |
| `export_global_grain_embedding.py` | One-time corpus **grain** PCA(3)+UMAP(3) for `latent_granular_3d_viewer` global embedding spaces |
| `build_latent_granular_3d_viewer.py` | Viewer + `bundles/manifest.json`; `--build-all-bundles` writes every codebook-folder × target JSON, or single pair via `--source-folder` + `--target-npy`. Optional `--global-embed-npz` / `--global-umap-model`. |
| `build_umap_structures_viewer.py` | Interactive 3D global UMAP: hulls, file graph, points, per-sound trajectories. |
| `export_umap_trajectories.py` | Export stacked UMAP coordinates per sound for external tools. |

---

## Decoding (not in these viewers)

All of the above live in **EnCodec latent space**. To **hear** the effect of moving along a curve or swapping grains, you need an **encode/decode** pipeline. The HTML viewers and PNGs are for **geometry and diagnostics** only unless you plug in the codec.

---

## Further reading

- Goodfire, *The world inside neural networks* — motivation for treating representations as **curved structure** and **paths**, not only linear directions: [goodfire.ai/research/the-world-inside-neural-networks](https://www.goodfire.ai/research/the-world-inside-neural-networks)
- *Latent Granular Resynthesis using Neural Audio Codecs* (ISMIR 2025 extended abstract) — algorithmic context for codebook–target matching and resynthesis: [arxiv.org/html/2507.19202v1](https://arxiv.org/html/2507.19202v1)
