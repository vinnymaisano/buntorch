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
        
        self.c_in = in_channels
        self.c_out = n_filters
        self.k = kernel_size
        self.padding = padding
        self.stride = stride

        self.x = None
        self.x_padded = None
        self.w_grad = None
        self.b_grad = None
    
    def compute_output_dim(self, d):
        return int((d - self.k + 2 * self.padding) / self.stride + 1)
    
    # loop implementation
    def __call__(self, x):
        # cache input for backprop
        self.x = x

        b, c, h, w = x.shape
        h_out = self.compute_output_dim(h)
        w_out = self.compute_output_dim(w)

        out = np.zeros((b, self.c_out, h_out, w_out))
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.x_padded = x_padded

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.k
                
                w_start = j * self.stride
                w_end = w_start + self.k

                # x_slice: [B, C_in, kernel_size, kernel_size]
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]

                # at position h_out, w_out: compute each filter across c_in
                for f in range(self.c_out):
                    # out[:, f, i, j]: [B, 1, 1, 1]
                    # w[f]: [C_in, kernel_size, kernel_size]
                    # (x_slice * w[f]).sum(axis=(1, 2, 3)): [B]
                    # b[f]: [B]
                    out[:, f, i, j] = (x_slice * self.w[f, :, :, :]).sum(axis=(1, 2, 3)) + self.b[f]
        return out

    def backward(self, out_grad):
        # out_grad shape: [b, n_filters, h_out, w_out]
        b, c_in, h_in, w_in = self.x.shape
        _, c_out, h_out, w_out = out_grad.shape
        self.w_grad = np.zeros((self.c_out, c_in, self.k, self.k))
        self.b_grad = np.zeros((self.c_out,))
        
        # in_grad = np.zeros((b, self.c_in, h_in, w_in))
        in_grad_padded = np.zeros((b, c_in, h_in + 2 * self.padding, w_in + 2 * self.padding))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.k

                w_start = j * self.stride
                w_end = w_start + self.k

                # Extract the exact input patch used during forward pass: (B, C_in, K, K)
                x_slice = self.x_padded[:, :, h_start:h_end, w_start:w_end]
                
                for f in range(c_out):
                    # Isolate the batch errors for this specific pixel and filter: (B,)
                    out_grad_slice = out_grad[:, f, i, j]
                    
                    # BIAS GRADIENT: Sum all batch errors passing through this filter channel
                    self.b_grad[f] += out_grad_slice.sum()
                    
                    # WEIGHT GRADIENT: Expand (B,) to (B,1,1,1) to broadcast multiply x_slice
                    # w_grad[f]: (B, ) - x_slice: (B, c_in, k, k) - out_grad_slice: (B, )
                    self.w_grad[f] += (x_slice * out_grad_slice[:, None, None, None]).sum(axis=0)
                    
                    # INPUT GRADIENT: Multiply the scalar errors by this filter's weights
                    # w[f] shape: (C_in, K, K)
                    # out_grad_slice[:, None, None, None] shape: (B, 1, 1, 1)
                    # Result shape: (B, C_in, K, K)
                    in_grad_padded[:, :, h_start:h_end, w_start:w_end] += (
                        self.w[f] * out_grad_slice[:, None, None, None]
                    )
        
        # 4. Remove padding borders to match the original unpadded input tensor 'x'
        if self.padding > 0:
            in_grad = in_grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            in_grad = in_grad_padded
        
        return in_grad
    
# TODO: fast convolution, implement pooling, train on MNIST, then start CIFAR
# will neeed regularization techniques
    
class Flatten:
    def __init__(self):
        self.original_shape = None

    def __call__(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, out_grad):
        return out_grad.reshape(self.original_shape)