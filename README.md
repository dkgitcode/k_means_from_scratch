# K Means Clustering from Scratch
A K-means clustering classifier built from scratch (no numpy or sk). 

## Theory
K-means clustering is an unsupervised machine learning algorithm that groups data points into k clusters based on euclidean distance. The algorithm works by placing "magnets" (centroids) in euclidean space and assigning each data point to the nearest magnet. The algorithm then moves the magnets to the center of their assigned data by simply averaging the data points. This process is repeated until the magnets no longer move.


## Fitting the data

```python
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
```

### Picking Magnets (Centroids)

To start, we randomly place our magnets on top of k data points. After the magnets are placed, data points that are closest to a magnet will now be assigned to that magnet. 

### Assigning points to a magnet (Clustering)

To define whether a point should be assigned to a magnet, we calculate the euclidean distance between the point and each magnet. The point is then assigned to the magnet with the smallest distance. 

### Euclidean distance formula

$$\sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$


### Moving the magnets

After all points are assigned to a magnet, we move the magnet to the center of the assigned points. This is done by simply averaging the points.

### Stopping the algorithm

The algorithm stops when the magnets no longer move. This is done by comparing the old magnet positions to the new magnet positions. If the magnets no longer move, the algorithm stops. However, we also have a max_iterations parameter to prevent the algorithm from running too long. 

## Predicting

Although not implemented, to predict a new data point, we would simply calculate the euclidean distance between the new data point and each magnet. Which ever magnet has the smallest distance would be the predicted cluster.





