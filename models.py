import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.w = np.random.randn(num_features, 1)
        self.b = np.random.randn(1)

    def __call__(self, x):
        return np.dot(x, self.w) + self.b
    
    def __repr__(self):
        return f"<LinearRegression n={self.w.shape[0]} w={self.w.reshape(-1)} b={self.b}>"
    
    def train(self, x, y, learning_rate=0.01, epochs=1000, verbose=False):
        m = x.shape[0]
        for epoch in range(epochs):
            y_pred = self.__call__(x)
            error = y_pred - y
            dw = (1/m) * np.dot(x.T, error)
            db = (1/m) * np.sum(error)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if verbose:print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(error**2)}")

class LogisticRegression:
    def __init__(self, num_features):
        self.w = np.random.randn(num_features, 1)
        self.b = np.random.randn(1)

    def __call__(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)
    
    def __repr__(self):
        return f"<LogisticRegression n={self.w.shape[0]} w={self.w.reshape(-1)} b={self.b}>"
    
    def train(self, x, y, learning_rate=0.01, epochs=1000, verbose=False):
        m = x.shape[0]
        for epoch in range(epochs):
            y_pred = self.__call__(x)
            error = y_pred - y
            dw = (1/m) * np.dot(x.T, error)
            db = (1/m) * np.sum(error)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if verbose:print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(error**2)}")
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))