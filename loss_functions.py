import numpy as np

class MSELoss:
    def __init__(self):
        self.ypred = None
        self.ytrue = None

    def __call__(self, ypred, ytrue):
        self.ypred = ypred
        self.ytrue = ytrue
        return np.mean((ypred-ytrue) ** 2)
    
    def backward(self, out_grad=1):
        m = self.ypred.shape[0]
        return (2 / m) * (self.ypred - self.ytrue) * out_grad