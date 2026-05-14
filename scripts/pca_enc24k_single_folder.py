import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


audio_folder = "data/enc24k/instrument_samples"
audio_file_name = "283142_2050105.npy"
audio_file_path = f"{audio_folder}/{audio_file_name}"

output_folder = "results/PCA_plots"


# create a low-dimensional representation of all slices combined 
# from all audio sources, tagged by folder and file name

# test with a single folder, "instrument_samples"
instrument_samples_folder = "data/enc24k/instrument_samples"
instrument_samples_files = [f for f in os.listdir(instrument_samples_folder) if f.endswith(".npy")]

# load all instrument samples
instrument_samples = [np.load(f"{instrument_samples_folder}/{f}") for f in instrument_samples_files]

# join all instrument samples into a single array, but keeping some kind of tag for the instrument source
instrument_samples_concatenated = []
instrument_source_tags = []
for i, f in enumerate(instrument_samples_files):
    instrument_samples_concatenated.append(instrument_samples[i])
    instrument_source_tags.append(np.full((instrument_samples[i].shape[0], 1), i))

instrument_samples_concatenated = np.concatenate(instrument_samples_concatenated, axis=0)
instrument_source_tags = np.concatenate(instrument_source_tags, axis=0)


# perform PCA on the concatenated instrument samples
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
instrument_samples_pca = pca.fit_transform(instrument_samples_concatenated)

# add the instrument source tags to the PCA results
instrument_samples_pca = np.concatenate((instrument_samples_pca, instrument_source_tags), axis=1)

# plot the PCA results, coloured by instrument source
# instead of the numeric tags, we will use the instrument name as a string
instrument_names = [f.split(".")[0] for f in instrument_samples_files]
instrument_names = np.array(instrument_names)


# Create a mapping from numeric tag to instrument name for each row in PCA result
instrument_indices = instrument_source_tags.flatten().astype(int)
instrument_names_per_row = instrument_names[instrument_indices]

# plot the PCA results, coloured by instrument name
plt.figure(figsize=(10, 10), dpi=300)
# Try "tab10", "Set2", and "Dark2" palettes; set a gray background
for palette in ["tab10", "Set2", "Dark2"]:
    plt.style.use('dark_background')
    ax = sns.scatterplot(
        x=instrument_samples_pca[:, 0],
        y=instrument_samples_pca[:, 1],
        hue=instrument_names_per_row,
        palette=palette
    )
    ax.set_facecolor('#444444')
    plt.title(f"PCA (palette: {palette})", fontsize=24)
    plt.savefig(f"{output_folder}/instrument_samples_pca_coloured_by_name_{palette}.png")
    plt.cla()  # clear for next plot
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20) 
# label the axes
plt.xlabel("PC1", fontsize=20)
plt.ylabel("PC2", fontsize=20)
plt.savefig(f"{output_folder}/instrument_samples_pca_coloured_by_name.png")
# plt.tight_layout()
plt.close()



