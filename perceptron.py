import numpy as np

class Perceptron:
    def __init__(self, 
                 learning_rate=1e-2,
                 n_iters=15,
                 ) -> None:
        self.lr = learning_rate
        self.n_iter = n_iters
        self.activation = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted  = self.activation(linear_output)

                #update
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_output)
        return y_predicted
    
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
if __name__ == "__main__":
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import make_classification

    # X, y = make_classification(n_features=4, n_redundant=0, n_clusters_per_class=1, n_samples=50)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y_train = np.array([
        0, 0, 0, 1
    ])

    p = Perceptron(learning_rate=0.01, n_iters=100)
    p.fit(X_train, y_train)
    predictions = p.predict(X_train)

    print("Predictions:", predictions)