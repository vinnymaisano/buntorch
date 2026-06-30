import numpy as np

class Tanh:
    def __init__(self):
        self.out = None

    def __call__(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, out_grad):
        return (1 - self.out ** 2) * out_grad
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def __call__(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, out_grad):
        return self.out * (1 - self.out) * out_grad
    
class ReLU:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, out_grad):
        return (self.x > 0) * out_grad