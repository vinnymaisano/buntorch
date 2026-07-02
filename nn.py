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

class ConvLayer:
    def __init__(self, in_channels, kernel_size, n_filters, padding, stride):
        fan_in = in_channels * kernel_size * kernel_size
        self.w = np.random.randn(n_filters, in_channels, kernel_size, kernel_size) / np.sqrt(2/fan_in)
        self.b = np.random.randn(n_filters)
        self.padding = padding
        self.stride = stride

        self.x = None
        self.w_grad = None
        self.b_grad = None
    
    # loop implementation
    def __call__(self, x):
        # cache input for backprop
        self.x = x

        b, c, h, w = x.shape
        h_out = self.compute_output_dim(h)
        w_out = self.compute_output_dim(w)

        out = np.zeros((b, self.n_filters, h_out, w_out))
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]

                # sum along filter dimension

                for f in range(self.n_filters):
                    # x_slice: [B, C_in, kernel_size, kernel_size]
                    # out[:, :, i, j]: [B, n_filters, h_out, w_out]
                    # w[f]: [C_in, kernel_size, kernel_size]
                    # (x_slice * w[f]).sum(axis=(1, 2, 3)): [B]
                    out[:, :, i, j] = (x_slice * self.w[f, :, :, :]).sum(axis=(1, 2, 3)) + self.b[f]
        return out
    
    def compute_output_dim(self, d):
        return int((d - self.kernel_size + 2 * self.padding) / self.stride + 1)