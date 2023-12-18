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

username = 'Jasper'
n_clusters = 9
plot_clusters = True
optimize = False

if username == 'Jasper':
    sample = [0.23297161802522615, 0.7362131426783016, 0.01]   
if username == 'Alissa':
    sample = [0.7434999126684984, 0.26637705972487136, 0.19744428269987505] 
             

#find average ratings for cute and scary
with open('ratings.csv', 'r') as file:
    ratings = list(csv.reader(file))[1:]

cute = [float(row[1]) for row in ratings if row[3] == username and row[2] == 'cute']
scary = [float(row[1]) for row in ratings if row[3] == username and row[2] == 'scary']

avg_cute = np.mean(cute, axis=0)
avg_scary = np.mean(scary, axis=0) 

# lower cute and lower scary means more likely to be binary or shapes
# higher cute and higher scary means more likely to be colors

shape_weight, color_weight, binary_weight = sample[0], sample[1], sample[2]
# shape_weight, color_weight, binary_weight = 0.0, 1.0, 0
image_dir = 'index/'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
binary_classification_path = 'beetle_binary_classifications.csv'

shape_scaled, colors_scaled, binary_scaled = process_images(image_paths, binary_classification_path)

user_labels = user_cluster_labels(username, n_clusters, random_state=10, plot=False)

shape_weighted = shape_scaled * shape_weight
color_weighted = colors_scaled * color_weight
binary_weighted = binary_scaled * binary_weight

combined_metrics = np.concatenate((shape_weighted, color_weighted, binary_weighted), axis=1)

#cluster the binary classes
kmeans = KMeans(n_clusters=n_clusters, random_state = 42069, n_init='auto').fit(combined_metrics)
predicted_labels = kmeans.labels_

# dbscan = DBSCAN(eps=0.01, min_samples=3).fit(combined_metrics)
# predicted_labels = dbscan.labels_
# print(predicted_labels)

if plot_clusters:
    grouped_images = {i: [] for i in range(n_clusters)}
    for img_path, label in zip(image_paths, predicted_labels):
        grouped_images[label].append(img_path)

    # Plot images in each cluster
    for cluster, img_paths in grouped_images.items():
        plt.figure(figsize=(10, 2))
        plt.suptitle(f'Cluster {cluster + 1}', fontsize=16)
        for i, img_path in enumerate(img_paths, start=1):
            img = Image.open(img_path)  # Open the image file
            plt.subplot(1, len(img_paths), i)
            plt.imshow(img)  # Display the image
            plt.axis('off')
        plt.show()


#compare the clusters to the binary classes
print(f"{sample}, nmi: {normalized_mutual_info_score(user_labels, predicted_labels)}")

if optimize:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    # Define the space of weights
    space  = [Real(0.01, 1.0, name='shape_weight'), 
            Real(0.01, 1.0, name='color_weight'), 
            Real(0.01, 1.0, name='binary_weight')]

    # Define the objective function
    @use_named_args(space)
    def objective(**params):
        shape_weight = params['shape_weight']
        color_weight = params['color_weight']
        binary_weight = params['binary_weight']

        # Your existing code to process images and perform clustering
        shape_weighted = shape_scaled * shape_weight
        color_weighted = colors_scaled * color_weight
        binary_weighted = binary_scaled * binary_weight

        combined_metrics = np.concatenate((shape_weighted, color_weighted, binary_weighted), axis=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42069, n_init='auto').fit(combined_metrics)
        predicted_labels = kmeans.labels_

        # Return the negative NMI score
        return -normalized_mutual_info_score(user_labels, predicted_labels)

    # Perform optimization
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

    # Best weights
    best_weights = res_gp.x
    print(f"Optimal weights: {best_weights}")

    # You can now use these weights in your clustering process
