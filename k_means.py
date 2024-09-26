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
    
    def fit(self, X):
        m = len(X)
        centroid_indices = random.sample(range(m), self.k) # fix
        self.centroids = [X[i] for i in centroid_indices]
        clusters = [[i] for i in centroid_indices]

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            
            # Assign clusters
            clusters = self.assign_clusters(X)
            
            # Update centroids
            self.centroids = self.get_centroids(X, clusters)
            
            # Check for convergence
            if all(old == new for old, new in zip(old_centroids, self.centroids)):
                break

        return clusters
    
    def find_elbow(self, ssd_values, k_values):
        """Find the elbow (maximum curvature) in the SSD plot using the distance from line method."""
        # Get the first and last points
        p1 = (k_values[0], ssd_values[0])
        p2 = (k_values[-1], ssd_values[-1])
        
        # Find the distances from each point to the line (p1 to p2)
        distances = []
        for i in range(len(k_values)):
            # Calculate the perpendicular distance from each point to the line
            x0, y0 = k_values[i], ssd_values[i]
            numerator = abs((p2[1] - p1[1]) * x0 - (p2[0] - p1[0]) * y0 + p2[0] * p1[1] - p2[1] * p1[0])
            denominator = math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
            distances.append(numerator / denominator)
        
        # The index with the maximum distance is the elbow
        max_distance_index = distances.index(max(distances))
        optimal_k = k_values[max_distance_index]
        return optimal_k

    def calculate_second_derivative(self, y):
        """Calculate the second derivative of y using pure Python."""
        first_derivative = [y[i+1] - y[i] for i in range(len(y) - 1)]
        second_derivative = [first_derivative[i+1] - first_derivative[i] for i in range(len(first_derivative) - 1)]
        return second_derivative

    def argmax(self, x):
        """Find the index of the maximum value in a list."""
        return max(range(len(x)), key=lambda i: x[i])

    def find_elbow_second_derivative(self, wcss_values, k_values):
        """Find the elbow using the second derivative method."""
        second_derivative = self.calculate_second_derivative(wcss_values)
        
        # The elbow is typically where the second derivative is maximum
        elbow_index = self.argmax(second_derivative) + 2  # +2 because of two derivative operations
        optimal_k = k_values[elbow_index]
        
        return optimal_k

    def fit_with_elbow(self, X, max_k=10):
        wcss_values = []
        k_values = list(range(1, max_k + 1))
        
        # Run K-means for each value of k and store WCSS
        possible_clusters = []
        for k in k_values:
            self.k = k  # Update k
            clusters = self.fit(X)
            wcss = self.get_wcss(X, clusters)
            wcss_values.append(wcss)
            possible_clusters.append(clusters)
        
        # Plot the elbow graph
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, wcss_values, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        
        # Calculate and plot second derivative
        second_derivative = self.calculate_second_derivative(wcss_values)
        plt.subplot(1, 2, 2)
        plt.plot(k_values[2:], second_derivative, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Second Derivative of WCSS')
        plt.title('Second Derivative Method')
        
        plt.tight_layout()
        plt.show()
        
        # Automatically find the elbow point using the second derivative method
        optimal_k = self.find_elbow_second_derivative(wcss_values, k_values)
        print(f"Optimal number of clusters (k) found: {optimal_k}")
        optimal_cluster = possible_clusters[optimal_k-1]
        
        # Return the optimal clusters
        return optimal_cluster
            