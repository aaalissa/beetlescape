"""Chinese Restaurant Process Infinite Mixture Model

This implementation assumes that the likelihood function and the prior distribution 
over the parameters are conjugate pairs.
"""

import random
import numpy
import scipy.stats as stats
import scipy.cluster.vq as vq
from skimage.color import deltaE_ciede2000
import cv2
from utils import get_image_paths
from sklearn.cluster import KMeans

def color_distance_ciede2000(lab_color1, lab_color2):
    return deltaE_ciede2000(lab_color1, lab_color2)

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

        #convert RGB to Lab
        pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    else:
        raise ValueError("Unexpected number of channels in image")

    pixels = numpy.float32(pixels)
    kmeans = KMeans(n_clusters=num_colors, n_init=1, max_iter=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

class CRPClusterModel(object):
	"""Chinese Restaurant Process Infinite Mixture Model

	Non-parametric Bayesian clustering with Chinese Restaurant Process prior

	The parameters for Gibbs sampling can be specified:
		num_iter : number of iterations to run. One iteration cycles through every data point once.
		eb_start : The trial where Empirical Bayes alpha adjustment begins
		eb_interval : The interval (number of trials) at which we adjust alpha
	"""

	def __init__(self, alpha, likelihood_fn):
		"""Initialize with the concentration hyperparameter alpha and likelihood function

		The likelihood function must be have this form
		def likelihood_fn(data, i, clustering, cluster_assn):
			Returns a vector x of length len(clustering) + 1 
			x[j] = P(data[i] | the cluster assignment so far AND data[i] assign to cluster j)

		where

		clustering - a list of clusters. Each cluster is a list of indices in the data
		cluster assignment - a list of cluster number (assignment)
			Examples
			Cluster 0 contains data from [1, 2, 5]
			Cluster 1 contains data from [0, 3, 4]

			Then clustering == [ [1,2,5], [0,3,4] ]
			AND cluster_assn = [1, 0, 0, 1, 1, 0]
			Note that the two formats are redundant.
		"""
		self.alpha = alpha
		self.likelihood_fn = likelihood_fn
		
		#gibbs sampling parameters
		self.num_iter = 100
		self.eb_start = 20
		self.eb_interval = 5

	def cluster(self, data):
		"""Cluster the data based on CRP prior infinite mixture model

		Args
		data must be a list of data points. Each data point can be any form.
		but self.likelihood_fn should be implemented accordingly.

		Returns 
			clustering - a list of clusters. Each cluster is a list of indices in the data
			cluster assignment - a list of cluster number (assignment)
		"""
		return self._gibbs_sampling_crp(data)

	def _initialize_assn(self, data):
		"""Initial cluster assignment before Gibbs sampling Process
		"""
		clustering = []
		cluster_assn = []

		for i in range(len(data)):
			crp_prior = [(len(x) + 0.0) / (i + self.alpha) for j, x in enumerate(clustering)]
			crp_prior.append(self.alpha / (i + self.alpha))
			crp_prior = numpy.array(crp_prior)
			likelihood = self.likelihood_fn(data, i, clustering, cluster_assn)
			probs = crp_prior * likelihood

			cluster = sample_with_weights(probs)
			if cluster == len(clustering):
				s = set([i])
				clustering.append(s)
			else:
				clustering[cluster].add(i)
			cluster_assn.append(clustering[cluster])
		return clustering, cluster_assn

	def _gibbs_sampling_crp(self, data):
		"""Run Gibbs sampling to get the cluster assignment """

		num_data = len(data)
		clustering, cluster_assn = self._initialize_assn(data)
		for t in range(self.num_iter):
			num_new_clusters = 0.0
			for i in range(num_data):
				cluster_assn[i].remove(i)
				if len(cluster_assn[i]) == 0:
					clustering.remove(cluster_assn[i])
				crp_prior = [(len(x) + 0.0) / (num_data - 1 + self.alpha) for j, x in enumerate(clustering)]
				crp_prior.append(self.alpha / (num_data - 1 + self.alpha))
				crp_prior = numpy.array(crp_prior)
				likelihood = self.likelihood_fn(data, i, clustering, cluster_assn)
				probs = crp_prior * likelihood

				cluster = sample_with_weights(probs)
				if cluster == len(clustering):
					s = set([i])
					clustering.append(s)
					num_new_clusters += 1
				else:
					clustering[cluster].add(i)
				cluster_assn[i] = clustering[cluster]
			# Empirical Bayes for adjusting hyperparameters
			if t % self.eb_interval == 0 and t > self.eb_start:
				self.alpha = num_new_clusters		
            
		return clustering, cluster_assn

def sample_with_weights(weights, sum_weights=None):
	"""Sample from a multinomial distribution

	Args:
		weights - a numpy array of positive numbers of associated weights for each index
		sum_weights - the sum of the above list. if we have call this function many times
			on the same weight, providing the sum will save a lot of computation time

	Returns:
		the index that gets chosen.
		-1 if a weight is invalid
	"""
	if sum_weights is None:
		sum_weights = numpy.sum(weights)
	p = random.uniform(0, sum_weights)
	sum_roulette = 0
	for i, weight in enumerate(weights):
		if weight < 0:
			return -1
		sum_roulette = sum_roulette + weight
		if (p < sum_roulette):
			return i
	return -1 

def example_likelihood_fn(data, i, clustering, cluster_assn):
	"""Example of likelihood function """
	means = [numpy.mean(data[list(cluster)]) for cluster in clustering]
	means.append(0)	
	stds = [1 for cluster in clustering]
	stds.append(10)
	return stats.norm.pdf(data[i], means, stds)


def color_likelihood_fn(data, i, clustering, cluster_assn):
    """
    Modified likelihood function based on color distance with a threshold.

    Args:
        data: List of color data points (in Lab color space).
        i: Index of the current data point.
        clustering: Current clustering (list of clusters, each cluster is a list of indices).
        cluster_assn: List of cluster assignments.

    Returns:
        A numpy array of likelihoods for each cluster.
    """
    threshold = 30  # Define a threshold for color distance
    likelihoods = []
    for cluster in clustering:
        if len(cluster) == 0:
            # Handle empty cluster case
            likelihoods.append(1)
            continue

        # Calculate the average color distance to the colors in the cluster
        cluster_colors = [data[j] for j in cluster]
        avg_distance = numpy.mean([color_distance_ciede2000(data[i], color) for color in cluster_colors])
    
        likelihood = numpy.exp( -(avg_distance) / 20)
        likelihoods.append(likelihood)

    # Handle new cluster case
    likelihoods.append(1)  # You might want to adjust this based on your assumptions

    return numpy.array(likelihoods)

if __name__ == '__main__':
    image_paths = get_image_paths()
    data = []
    for image in image_paths:
        color = extract_colors(image, num_colors=1).flatten().tolist()
        data.append(tuple(color))
    #random.shuffle(data)
    crp_model = CRPClusterModel(0.5, color_likelihood_fn)
    clustering, cluster_assn = crp_model.cluster(data)

        #remove duplicates
    cluster_tuples = [tuple(sorted(cluster)) for cluster in cluster_assn]

    # Remove duplicates by converting the list to a set and back to a list
    unique_cluster_tuples = list(set(cluster_tuples))

    # If you need to convert back to a list of sets
    unique_clusters = [set(cluster) for cluster in unique_cluster_tuples]

    print(f'clustering assignment: {unique_clusters}')