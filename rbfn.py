import numpy as np

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

class kmeans:
    def __init__(self, values, number_of_clusters = 2, random_state = 0):
        try:
            self.values = values
            self.number_of_clusters = number_of_clusters
            self.random_state = random_state
            self.clf = KMeans(n_clusters=number_of_clusters,random_state=self.random_state).fit(self.values)
        except Exception as e:
            print("Ocorreu um erro na classificacao do kmeans")
            print(str(e))
            exit(0)
  

    def get_cluster_centers(self):
        return self.clf.cluster_centers_

class rbf:
    def __init__(self, number_of_inputs, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
    
    def _activation_function(self, x, weights, sigma):
        #Gaussian function
        try:
            return np.exp(-np.sum((x-weights)**2, axis=1) / (2 * sigma ** 2))
        except Exception as e:
            print("Ocorreu um erro na função de ativação")
            print(str(e))
            print("x = ",x)
            print("weights = ", weights)
            print("sigma=  ", sigma)
            exit(0)
    
    def fit(self, X, number_of_clusters):
        kmeansClf = kmeans(X, number_of_clusters)
        centers = kmeansClf.get_cluster_centers()

        a = [None] * len(centers)
        for index, center in enumerate(centers):
            a[index] = self._activation_function(X, center, X.std())

        return dict(output=np.array(a), weights=centers)


class RBFNetwork:

    def __init__(self, size_of_inputs, number_of_clusters, number_of_hidden_layers=10, learning_rate=0.1):
        
        self.size_of_inputs = size_of_inputs
        self.number_of_clusters = number_of_clusters
        self.number_of_hidden_layers = number_of_hidden_layers
        self.learning_rate = learning_rate
        
        self.rbf = rbf(self.size_of_inputs, self.number_of_clusters)
        self.mlp = MLPClassifier(hidden_layer_sizes=(number_of_hidden_layers), alpha=1e-4,
                    solver='sgd', verbose=True, tol=1e-4, random_state=1,
                    learning_rate_init=self.learning_rate)


        self.weights = None
  
    
  
    def fit(self, training, targets):
        result = self.rbf.fit(training, self.number_of_clusters)
        self.weights = result['weights']
        self.mlp.fit(result['output'], targets)

    def predict(self, test):
        y_pred = [None] * len(self.weights)
        try:    
            for index, weight in enumerate(self.weights):
                y_pred[index] = self.rbf._activation_function(test, weight, test.std())
            y_pred = np.array(y_pred)
            return self.mlp.predict(np.transpose(y_pred))
        except Exception as e:
            print("Ocorreu um erro na função de predição do RBFN")
            print("Exceção: ", str(e))
            print(y_pred)
            exit(0)


if __name__ == "__main__":
    xor_entry_values = np.array([[0, 0], [0 , 1], [1, 0], [1,1]])
    xor_targets = np.array([0, 1, 1, 0])
    
    rbfn = RBFNetwork(2, 4)
    rbfn.fit(xor_entry_values, xor_targets)

    test = np.array([[1,0], [0,1]])
    result = rbfn.predict(test)
    print(result)