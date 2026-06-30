import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def get_params(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.extend(layer.get_params())
        return params
    
    def get_param_count(self):
        count = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                count += layer.get_param_count()
        return count
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, out_grad=1):
        in_grad = out_grad
        for layer in reversed(self.layers):
            in_grad = layer.backward(in_grad)
        return in_grad

class Linear:
    def __init__(self, n_in, n_out):
        self.w = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        self.b = np.random.randn(n_out)
        self.x = None
        self.w_grad = 0
        self.b_grad = 0

    def get_params(self):
        return [self.w, self.b]
    
    def get_param_count(self):
        return self.w.size + self.b.size

    def __call__(self, x):
        self.x = x
        return x @ self.w + self.b
    
    def __repr__(self):
        return f"<Linear n_in={self.w.shape[0]} n_out={self.w.shape[1]}>"
    
    def backward(self, out_grad):
        self.w_grad = self.x.T @ out_grad
        self.b_grad = np.sum(out_grad, axis=0)
        in_grad = out_grad @ self.w.T
        return in_grad

def zero_pad(x, padding):
    pass

class Convolution:
    def __init__(self, in_channels, kernel_size, n_filters, padding, stride):
        fan_in = in_channels * kernel_size * kernel_size
        self.w = np.random.randn(n_filters, in_channels, kernel_size, kernel_size) / np.sqrt(2/fan_in)
        self.b = np.random.randn(n_filters)
        self.padding = padding
        self.stride = stride

        self.x = None
        self.w_grad = None
        self.b_grad = None
    
    def __call__(self, x):
        b, c, h, w = x.shape

        x_pad = np.zeros(h + 2*self.padding, w + 2*self.padding)