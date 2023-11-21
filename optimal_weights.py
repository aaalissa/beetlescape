# Import Statements
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Local imports
from utils import process_images, user_cluster_labels

# Constants and Configuration
USERNAME = 'Jasper'
N_CLUSTERS = 7
PLOT_CLUSTERS = False
PLOT_USER = True
IMAGE_DIR = 'index/'
BINARY_CLASSIFICATION_PATH = 'beetle_binary_classifications.csv'
RANDOM_STATE_KMEANS = 42069
RANDOM_STATE_OPTIMIZATION = 0

# Data Loading and Preprocessing
def load_ratings(username):
    with open('ratings.csv', 'r') as file:
        ratings = list(csv.reader(file))[1:]
    cute = [float(row[1]) for row in ratings if row[3] == username and row[2] == 'cute']
    scary = [float(row[1]) for row in ratings if row[3] == username and row[2] == 'scary']
    return np.mean(cute), np.mean(scary)

def load_and_process_images():
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return process_images(image_paths, BINARY_CLASSIFICATION_PATH)

# Clustering and Analysis
def perform_clustering(shape_weight, color_weight, binary_weight):
    shape_scaled, colors_scaled, binary_scaled = load_and_process_images()
    combined_metrics = np.concatenate((shape_scaled * shape_weight, 
                                       colors_scaled * color_weight, 
                                       binary_scaled * binary_weight), axis=1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE_KMEANS, n_init='auto').fit(combined_metrics)
    return kmeans.labels_

def plot_clusters(predicted_labels, image_paths):
    if PLOT_CLUSTERS:
        grouped_images = {i: [] for i in range(N_CLUSTERS)}
        for img_path, label in zip(image_paths, predicted_labels):
            grouped_images[label].append(img_path)

        for cluster, img_paths in grouped_images.items():
            plt.figure(figsize=(10, 2))
            plt.suptitle(f'Cluster {cluster + 1}', fontsize=16)
            for i, img_path in enumerate(img_paths, start=1):
                img = Image.open(img_path)
                plt.subplot(1, len(img_paths), i)
                plt.imshow(img)
                plt.axis('off')
            plt.show()

# Optimization
def objective(shape_weight, color_weight, binary_weight):
    predicted_labels = perform_clustering(shape_weight, color_weight, binary_weight)
    user_labels = user_cluster_labels(USERNAME, N_CLUSTERS, random_state=10, plot=PLOT_USER)
    return -normalized_mutual_info_score(user_labels, predicted_labels)

space = [Real(0.01, 1.0, name='shape_weight'), 
         Real(0.01, 1.0, name='color_weight'), 
         Real(0.01, 1.0, name='binary_weight')]

@use_named_args(space)
def objective_wrapper(**params):
    return objective(**params)

res_gp = gp_minimize(objective_wrapper, space, n_calls=50, random_state=RANDOM_STATE_OPTIMIZATION)
best_weights = res_gp.x
print(f"Optimal weights: {best_weights}")

# Main Execution
if __name__ == "__main__":
    avg_cute, avg_scary = load_ratings(USERNAME)
    predicted_labels = perform_clustering(*best_weights)
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    plot_clusters(predicted_labels, image_paths)
    print(f"Best weights: {best_weights}, NMI: {objective(*best_weights)}")
