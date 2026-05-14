import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


enc_root = "data/enc24k"
output_folder = "results/PCA_plots"
os.makedirs(output_folder, exist_ok=True)

# One marker style per top-level folder under enc24k (7 folders)
FOLDER_MARKERS = ["o", "s", "^", "D", "P", "*", "X"]

subfolders = sorted(
    name
    for name in os.listdir(enc_root)
    if os.path.isdir(os.path.join(enc_root, name))
)
if len(subfolders) != 7:
    raise RuntimeError(
        f"expected 7 folders under {enc_root}, got {len(subfolders)}: {subfolders}"
    )
if len(FOLDER_MARKERS) < len(subfolders):
    raise RuntimeError("FOLDER_MARKERS list is shorter than number of folders")

folder_to_marker = {folder: FOLDER_MARKERS[i] for i, folder in enumerate(subfolders)}

blocks = []
sound_labels = []
folder_labels = []

for folder in subfolders:
    folder_path = os.path.join(enc_root, folder)
    for fname in sorted(f for f in os.listdir(folder_path) if f.endswith(".npy")):
        arr = np.load(os.path.join(folder_path, fname))
        stem = os.path.splitext(fname)[0]
        n_rows = arr.shape[0]
        blocks.append(arr)
        sound_labels.extend([f"{folder}/{stem}"] * n_rows)
        folder_labels.extend([folder] * n_rows)

X = np.concatenate(blocks, axis=0)
sound_labels = np.array(sound_labels)
folder_labels = np.array(folder_labels)

unique_sounds = np.unique(sound_labels)
n_sounds = len(unique_sounds)
palette = sns.color_palette("husl", n_colors=n_sounds)
sound_to_color = dict(zip(unique_sounds, palette))

plt.style.use("dark_background")


def plot_2d(
    Y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    sns.scatterplot(
        x=Y[:, 0],
        y=Y[:, 1],
        hue=sound_labels,
        style=folder_labels,
        markers=folder_to_marker,
        palette=palette,
        s=24,
        alpha=0.75,
        ax=ax,
        edgecolor="none",
    )
    ax.set_facecolor("#444444")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title="sound / folder",
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d(
    Y: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(14, 12), dpi=300)
    ax3 = fig.add_subplot(projection="3d")
    for sound in unique_sounds:
        mask = sound_labels == sound
        folder = folder_labels[mask][0]
        ax3.scatter(
            Y[mask, 0],
            Y[mask, 1],
            Y[mask, 2],
            color=sound_to_color[sound],
            marker=folder_to_marker[folder],
            s=24,
            alpha=0.75,
            edgecolors="none",
            label=sound,
        )
    ax3.set_facecolor("#444444")
    ax3.set_title(title, fontsize=20, pad=20)
    ax3.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax3.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax3.set_zlabel(zlabel, fontsize=14, labelpad=10)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor("#666666")
    ax3.yaxis.pane.set_edgecolor("#666666")
    ax3.zaxis.pane.set_edgecolor("#666666")
    ax3.legend(
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        fontsize=7,
        title="sound / folder",
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- PCA ---
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X)
plot_2d(
    X_pca_3d[:, :2],
    "PC1",
    "PC2",
    "PCA enc24k — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_pca_by_sound.png",
)
plot_3d(
    X_pca_3d,
    "PC1",
    "PC2",
    "PC3",
    "PCA enc24k (3D) — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_pca_by_sound_3d.png",
)

# --- UMAP ---
umap_reducer = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
X_umap_3d = umap_reducer.fit_transform(X)

folder_to_id = {name: i for i, name in enumerate(subfolders)}
sound_to_id = {s: i for i, s in enumerate(unique_sounds)}
folder_ids = np.array([folder_to_id[f] for f in folder_labels], dtype=np.int16)
sound_ids = np.array([sound_to_id[s] for s in sound_labels], dtype=np.int16)
rgb = np.array([sound_to_color[s] for s in sound_labels], dtype=np.float32)
np.savez_compressed(
    f"{output_folder}/enc24k_umap_3d_export.npz",
    xyz=X_umap_3d.astype(np.float32),
    rgb=rgb,
    folder_id=folder_ids,
    sound_id=sound_ids,
    folders=np.array(subfolders),
    sounds=unique_sounds,
)

plot_2d(
    X_umap_3d[:, :2],
    "UMAP 1",
    "UMAP 2",
    "UMAP enc24k — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_umap_by_sound.png",
)
plot_3d(
    X_umap_3d,
    "UMAP 1",
    "UMAP 2",
    "UMAP 3",
    "UMAP enc24k (3D) — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_umap_by_sound_3d.png",
)

# --- t-SNE (PCA preprocess to cut noise; speeds up / stabilises on high-D audio embeddings) ---
X_pre_tsne = PCA(n_components=50, random_state=42).fit_transform(X)
tsne = TSNE(
    n_components=3,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42,
    verbose=1,
)
X_tsne_3d = tsne.fit_transform(X_pre_tsne)
plot_2d(
    X_tsne_3d[:, :2],
    "t-SNE 1",
    "t-SNE 2",
    "t-SNE enc24k — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_tsne_by_sound.png",
)
plot_3d(
    X_tsne_3d,
    "t-SNE 1",
    "t-SNE 2",
    "t-SNE 3",
    "t-SNE enc24k (3D) — colour = sound (per folder), marker = folder",
    f"{output_folder}/enc24k_all_folders_tsne_by_sound_3d.png",
)
