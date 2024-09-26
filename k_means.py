import random
import math
import matplotlib.pyplot as plt


class K_Means_Classifier:
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        
    def euclidean_distance(self, x1, x2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def argmin(self, x):
        _min = 0
        for i in range(len(x)):
            if x[i] < x[_min]:
                _min = i
        return _min
    
    def get_centroids(self,X, clusters):
        num_features = len(X[0])
        centroids = [[0.0 for _ in range(num_features)] for _ in range(len(clusters))]
        cluster_lengths = []
        for ci,point_indices in enumerate(clusters): 
            cluster_lengths.append(len(clusters[ci]))
            for point_index in point_indices:
                for i,feature in enumerate(X[point_index]): 
                    centroids[ci][i] += feature * 1/len(clusters[ci])
        return centroids
    
    def get_wcss(self, X, clusters):
        ssd = 0
        for cluster_index, cluster in enumerate(clusters):
            for point_index in cluster:
                ssd += self.euclidean_distance(X[point_index], self.centroids[cluster_index]) ** 2
        return ssd
    
    def assign_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for i, point in enumerate(X):
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            best_cluster = distances.index(min(distances))
            clusters[best_cluster].append(i)
        return clusters
    
    def fit(self, X, k=None):
        m = len(X)
        self.k = k if k else self.k
        centroid_indices = random.sample(range(m), self.k)
        self.centroids = [X[i] for i in centroid_indices]

        clusters = [[i] for i in centroid_indices]

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            clusters = self.assign_clusters(X)
            self.centroids = self.get_centroids(X, clusters)

            if all(old == new for old, new in zip(old_centroids, self.centroids)):
                break

        return clusters, self.centroids

    def argmax(self, x):
        """Find the index of the maximum value in a list."""
        return max(range(len(x)), key=lambda i: x[i])

    def plot_elbow(self, X, max_k=10):
        wcss_values = []
        k_values = list(range(1, max_k + 1))
        
        for k in k_values:
            self.k = k  # Update k
            clusters, _ = self.fit(X)
            wcss = self.get_wcss(X, clusters)
            wcss_values.append(wcss)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, wcss_values, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        
        
            