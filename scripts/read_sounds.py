import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


audio_folder = "data/enc24k/instrument_samples"
audio_file_name = "283142_2050105.npy"
audio_file_path = f"{audio_folder}/{audio_file_name}"

output_folder = "results/2D_plots"


a = np.load(audio_file_path)
print(audio_file_name, a.shape)

# each audio has 128 rows and N columns

num_slices = a.shape[0]
num_embeddings = a.shape[1]

print(f"this audio ({audio_file_name}) has {num_slices} slices and {num_embeddings} embeddings")

# choose a random slice
random_slice = np.random.randint(0, num_slices)
print(f"the shape of the {random_slice}th slice is {a[random_slice].shape}")



# let's plot a scatter plot of embeddings for the all slices combined
# we colour by slice index
sns.set_style("white")
plt.figure(figsize=(10, 10), dpi=300)
sns.scatterplot(
    x=a[:, 0], y=a[:, 1], 
    hue=np.arange(num_slices), 
    palette="viridis", 
    legend="brief", 
    hue_norm=(0, num_slices - 1)
)
plt.savefig(f"{output_folder}/embeddings_all.png")
plt.close()

