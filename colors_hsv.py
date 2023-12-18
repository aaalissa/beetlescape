import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Variables
directory = 'index/'
plot_bugs = True
n_bug_clusters = 8

def extract_colors(image_path, num_colors=1):
    # Read the image with alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert BGR to HSV (or BGRA to HSVA)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif image.shape[2] == 4:
        # Convert BGRA to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        # Create a mask to identify non-transparent pixels
        mask = image[:, :, 3] != 0

        # Apply the mask to get pixels that are not transparent
        pixels = image[mask][:, :3]

        # Convert RGB to HSV
        pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    else:
        raise ValueError("Unexpected number of channels in image")

    pixels = np.float32(pixels)
    kmeans = KMeans(n_clusters=num_colors, n_init=1, max_iter=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_



# Process each image and collect colors
image_color_profiles = {}

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file types if needed
        path = os.path.join(directory, filename)
        color_profile = extract_colors(path)   
        image_color_profiles[filename] = color_profile


if plot_bugs:
    for key, value in list(image_color_profiles.items()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        fig.suptitle(key)

        image = cv2.imread(directory + str(key))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax1.imshow(image)
        ax1.axis('off')

        # Convert HSV to RGB for display
        rgb_colors = cv2.cvtColor(value.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB)
        ax2.imshow(rgb_colors)
        ax2.axis('off')

        plt.show()


clustering_colors = []
for key, value in list(image_color_profiles.items()):
    clustering_colors.append(value.flatten())

clustering_colors = np.array(clustering_colors)
print(clustering_colors.shape)

kmeans = KMeans(n_clusters=n_bug_clusters, n_init=1, max_iter=100)
kmeans.fit(clustering_colors)
labels = kmeans.labels_

for i, key in enumerate(list(image_color_profiles.keys())):
    image_color_profiles[key] = labels[i]

# Group images based on cluster labels
image_clusters = {}
for i, key in enumerate(list(image_color_profiles.keys())):
    cluster_label = labels[i]
    if cluster_label not in image_clusters:
        image_clusters[cluster_label] = []
    image_clusters[cluster_label].append(key)


# Create a single plot with subplots for each cluster
for cluster_label, image_names in image_clusters.items():
    fig, axes = plt.subplots(1, len(image_names), figsize=(10, 4))
    fig.suptitle(f"Cluster {cluster_label}")

    # Ensure axes is always a list for consistency
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, image_name in enumerate(image_names):
        image_path = os.path.join(directory, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_name)

    plt.show()
