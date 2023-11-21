import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def normalize_filled(image_path, size=(150, 150)):
    # Read the image with alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel")

    # Use the alpha channel as a mask
    alpha_channel = image[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    bounding_rect = cv2.boundingRect(mask)
    img_cropped_bounding_rect = mask[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    img_resized = cv2.resize(img_cropped_bounding_rect, size)

    # Find contours on the mask
    contours, _ = cv2.findContours(img_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(img_resized, pts=contours, color=(255,255,255))

    return img_resized

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


def process_images(image_paths, binary_classification_path):
    # Initialize PCA and MinMaxScaler
    pca = PCA(n_components=5)
    scaler = MinMaxScaler()

    processed_images = [normalize_filled(path) for path in image_paths]

    # Initialize the similarity matrix
    n = len(processed_images)
    shape_matrix = np.zeros((n, n))

    # Fill the matrix with similarity scores
    for i in range(n):
        for j in range(n):
            if i != j:
                shape_matrix[i, j] = cv2.matchShapes(processed_images[i], processed_images[j], cv2.CONTOURS_MATCH_I1, 0.0)
    shape_pca = pca.fit_transform(shape_matrix)
    shape_scaled = scaler.fit_transform(shape_pca)

    colors = [extract_colors(path) for path in image_paths]
    colors = np.array(colors).reshape(-1, 3)
    colors_scaled = scaler.fit_transform(colors)

    with open(binary_classification_path, 'r') as file:
        binary_classes = list(csv.reader(file))[1:]  # remove header

    # Turn into np array and get rid of the first column
    binary_classes = np.array(binary_classes, dtype=float)[:, 1:]

    binary_pca = pca.fit_transform(binary_classes)
    binary_scaled = scaler.fit_transform(binary_pca)

    return shape_scaled, colors_scaled, binary_scaled



def user_cluster_labels(username, k, plot=True, random_state=0):
    """
    Performs k-means clustering on the user's coordinates and plots the results
    :param username: the username of the user
    :param k: the number of clusters
    :return: the labels of the clusters
    """
    # Load coordinates
    with open('coordinates_log.csv', 'r') as file:
        bugs_coords = list(csv.reader(file))[1:]  # remove header

    # Load ratings
    with open('ratings.csv', 'r') as file:
        ratings = list(csv.reader(file))[1:]

    # Extract user coordinates
    user_coords = [list(eval(row[3])) for row in bugs_coords if row[1] == username]
    for coord in user_coords:
        coord[1] = -coord[1]  # Flip y axis

    # Convert to numpy array for clustering
    user_coords = np.array(user_coords)

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(user_coords)
    labels = kmeans.labels_

    # Plotting
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(user_coords[:, 0], user_coords[:, 1], c=labels, cmap='Spectral')

        # Add annotations
        for i, txt in enumerate(user_coords):
            label = i + 1
            for row in ratings:
                if row[3] == username and eval(row[0]) == label:
                    category = row[2]
                    rating = row[1]
                    plt.annotate(f"id: {label}, {category}: {rating}", (user_coords[i, 0], user_coords[i, 1]))

    return labels