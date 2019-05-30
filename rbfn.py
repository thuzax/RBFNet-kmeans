import numpy as np

from sklearn.cluster import KMeans

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

class RBFNetwork:

    def __init__(self, centers, epochs=100, learning_rate=0.1):
        self.centers = centers
        self.number_of_clusters = len(self.centers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.weights = np.random.randn(self.number_of_clusters)
        self.bias = np.random.randn(1)
  
    def _activation_function(self, x, center):
        #Gaussian function
        try:
            distance = np.linalg.norm(x - center)
            return np.exp(-(distance ** 2))
        except Exception as e:
            print("Ocorreu um erro na função de ativação")
            print(str(e))
            print("x = ",x)
            print("center = ", center)
            exit(0)
  
    def fit(self, training, targets):
        for epoch in range(self.epochs):
            for training_index, training_instance in enumerate(training):
                a = []
                for index, center in enumerate(self.centers):
                    a.append(self._activation_function(training_instance, center))
                
                a = np.array(a)
                F = a.T.dot(self.weights) + self.bias

                loss = (targets[training_index] - F).flatten() ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(targets[training_index] - F).flatten()

                # online update
                self.weights = self.weights - self.learning_rate * a * error
                self.bias = self.bias - self.learning_rate * error

    def predict(self, test):
        y_pred = []
        for test_index, test in enumerate(test):
            a = []
            for index, center in enumerate(self.centers):
                a.append(self._activation_function(test, center))

            a = np.array(a)
            F = a.T.dot(self.weights) + self.bias
            y_pred.append(F)
        return np.array(y_pred)


if __name__ == "__main__":
    xor_entry_values = np.array([[0, 0], [0 , 1], [1, 0], [1,1]])
    xor_targets = np.array([[1], [0], [0], [1]])
    
    kmeans = kmeans(xor_entry_values)
    centers = kmeans.get_cluster_centers()
    rbf = RBFNetwork(centers)
    rbf.fit(xor_entry_values, xor_targets)

    print(rbf.predict([[0,0], [0,1], [1,0], [1,1]]))