
import numpy as np
import matplotlib.pyplot as plt
import math

##
def rbf(x, c, s):
    r = []
    for i in zip(x[0], x[1]):
        r.append(np.exp(- math.sqrt((i[0]-c[0])**2 + (i[1]-c[1])**2) **2))
    r = np.array(r)
    return r
    # return np.exp(-1 / (2 * s**2) * (x-c)**2)

##

def eucli_dist(x1, y1, x2, y2):
    x = x1 - x2
    y = y1 - y2
    return math.sqrt(x**2 + y**2)

def mean_point(x1s, x2s):
    sum_x1 = 0
    for x1 in x1s:
        sum_x1 += x1

    mean_x1 = sum_x1 / len(x1s)

    sum_x2 = 0
    for x2 in x2s:
        sum_x2 += x2

    mean_x2 = sum_x2 / len(x2s)

    return (mean_x1, mean_x2)

def std_point(x1s, x2s):
    mean_points = (mean_point(x1s, x2s))

    sum_x1 = 0
    for x1 in x1s:
        sum_x1 += (x1 - mean_points[0]) ** 2
    
    std_x1 = math.sqrt(sum_x1/len(x1s))


    sum_x2 = 0
    for x2 in x2s:
        sum_x2 += (x2 - mean_points[1]) ** 2

    std_x2 = math.sqrt(sum_x2/len(x2s))

    return(std_x1, std_x2)

def kmeans(X, k):
    """Performs k-means clustering for 1D input

    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters

    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    X1 = np.array(X[0])
    X2 = np.array(X[1])
    clusters_pos = np.random.choice(range(len(X1)), size=k)


    clusters = np.array([np.array((X1[i], X2[i])) for i in clusters_pos])
    prevClusters = clusters.copy()
    
    stds = []
    for i in zip(np.zeros(k), np.zeros(k)):
        stds.append(i)
    np.array(stds)

    converged = False


    while not converged:
        """
        compute distances for each cluster center to each point
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        # distances = np.squeeze(np.abs(X1[:, np.newaxis] - clusters[np.newaxis, :]))

        d_temp = []
        for x1, x2, in zip(X1, X2):
            line = []
            for c1, c2 in clusters:
                line.append(eucli_dist(x1,x2,c1,c2))
            d_temp.append(line)
        
        distances = np.array(d_temp)

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForClusterX1 = X1[closestCluster == i]
            pointsForClusterX2 = X2[closestCluster == i]
            if len(pointsForClusterX1) > 0:
                clusters[i] = mean_point(pointsForClusterX1, pointsForClusterX2)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()


    # distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    d_temp = []
    for x1, x2, in zip(X1, X2):
        line = []
        for c1, c2 in clusters:
            line.append(eucli_dist(x1,x2,c1,c2))
        d_temp.append(line)
    distances = np.array(d_temp)

    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForClusterX1 = X1[closestCluster == i]
        pointsForClusterX2 = X2[closestCluster == i]
        if len(pointsForClusterX1) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = std_point(pointsForClusterX1, pointsForClusterX2)
            # stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverageX1 = []
        pointsToAverageX2 = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverageX1.append(X1[closestCluster == i])
                pointsToAverageX2.append(X2[closestCluster == i])
        pointsToAverageX1 = np.concatenate(pointsToAverageX1).ravel()
        pointsToAverageX2 = np.concatenate(pointsToAverageX2).ravel()

        std_x1, std_x2 = std_point(pointsToAverageX1, pointsToAverageX2)
        stds[clustersWithNoPoints] = mean_point(std_x1, std_x2)
        # stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    print(clusters)
    print(stds)

    return clusters, stds

##

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    ##

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            # self.centers, self.stds = kmeans(X, self.k)
            self.centers, self.stds = [(0,0), (1, 1)], [(0, 0), [(0, 0)]]
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.stds)])
                
                F = a.T.dot(self.w) + self.b
                loss = (y[i] - F).flatten() ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()


                # online update

                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
    ##

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)


# sample inputs and add noise
NUM_SAMPLES = 4
# X = np.random.uniform(0., 1., NUM_SAMPLES)
# X = np.sort(X, axis=0)
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])


print(X)
# noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
# print(noise)
y = np.array([0, 1, 1, 0])

rbfnet = RBFNet(lr=1e-2, k=2)
rbfnet.fit(X, y)

y_pred = rbfnet.predict(X)

plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()

print(rbfnet.predict(X))