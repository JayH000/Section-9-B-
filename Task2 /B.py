import numpy as np

class CNN:
    def __init__(self, input_dim, conv_filter_size, num_filters, pool_size, fc_units, output_classes, activation='relu', lr=0.01):
        """
        Initializes CNN with:
        - Convolution layer (single layer with multiple filters)
        - Pooling layer (2x2 max pooling)
        - Fully connected layer (MLP-like structure)
        """
        self.input_dim = input_dim  # (height, width, channels)
        self.conv_filter_size = conv_filter_size
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.fc_units = fc_units
        self.output_classes = output_classes
        self.activation = activation
        self.lr = lr  # Learning rate

        # Initialize filters (random small values)
        self.filters = np.random.randn(num_filters, conv_filter_size, conv_filter_size, input_dim[2]) * 0.1
        self.biases = np.zeros((num_filters, 1))

        # Compute dimensions after convolution & pooling
        conv_output_size = input_dim[0] - conv_filter_size + 1  # Assuming stride = 1
        pool_output_size = conv_output_size // pool_size

        self.flatten_size = pool_output_size * pool_output_size * num_filters  # Size after flattening
        
        # Fully connected layer weights & biases
        self.W_fc = np.random.randn(fc_units, self.flatten_size) * 0.1
        self.b_fc = np.zeros((fc_units, 1))
        
        self.W_out = np.random.randn(output_classes, fc_units) * 0.1
        self.b_out = np.zeros((output_classes, 1))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def convolve(self, X):
        """Convolution operation."""
        H, W, C = X.shape
        F, HF, WF, _ = self.filters.shape  # (num_filters, filter_height, filter_width, channels)
        output_size = H - HF + 1  # Assuming stride = 1
        feature_maps = np.zeros((F, output_size, output_size))
        
        for f in range(F):
            for i in range(output_size):
                for j in range(output_size):
                    region = X[i:i+HF, j:j+WF, :]
                    feature_maps[f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        
        return feature_maps

    def pool(self, X):
        """Max Pooling operation."""
        F, H, W = X.shape
        pool_H, pool_W = H // self.pool_size, W // self.pool_size
        pooled = np.zeros((F, pool_H, pool_W))
        
        for f in range(F):
            for i in range(pool_H):
                for j in range(pool_W):
                    region = X[f, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    pooled[f, i, j] = np.max(region)
        
        return pooled

    def forward(self, X):
        """Forward pass."""
        self.X = X  # Store input for backpropagation

        # Convolution layer
        self.conv_out = self.convolve(X)
        self.activated_conv = self.relu(self.conv_out) if self.activation == 'relu' else self.sigmoid(self.conv_out)

        # Pooling layer
        self.pooled_out = self.pool(self.activated_conv)
        self.flattened = self.pooled_out.flatten().reshape(-1, 1)

        # Fully connected layer
        self.fc_out = np.dot(self.W_fc, self.flattened) + self.b_fc
        self.activated_fc = self.relu(self.fc_out) if self.activation == 'relu' else self.sigmoid(self.fc_out)

        # Output layer (classification)
        self.out = np.dot(self.W_out, self.activated_fc) + self.b_out
        self.probs = self.softmax(self.out)

        return self.probs

    def softmax(self, x):
        """Softmax function for multi-class classification."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    def backward(self, y_true):
        """Backward propagation."""
        # Compute gradients for output layer
        dL_dOut = self.probs - y_true  # Cross-entropy loss derivative

        dL_dW_out = np.dot(dL_dOut, self.activated_fc.T)
        dL_db_out = np.sum(dL_dOut, axis=1, keepdims=True)

        # Fully connected layer
        dL_dFc = np.dot(self.W_out.T, dL_dOut) * (self.relu_derivative(self.fc_out) if self.activation == 'relu' else self.sigmoid_derivative(self.activated_fc))

        dL_dW_fc = np.dot(dL_dFc, self.flattened.T)
        dL_db_fc = np.sum(dL_dFc, axis=1, keepdims=True)

        # Backpropagate to pooled layer (not updating pooling layer since it has no weights)
        dL_dFlattened = np.dot(self.W_fc.T, dL_dFc).reshape(self.pooled_out.shape)

        # Backpropagate to convolution layer
        dL_dConv = np.zeros_like(self.conv_out)
        for f in range(self.num_filters):
            dL_dConv[f] = np.kron(dL_dFlattened[f], np.ones((self.pool_size, self.pool_size)))  # Reverse pooling operation

        dL_dConv *= self.relu_derivative(self.conv_out) if self.activation == 'relu' else self.sigmoid_derivative(self.conv_out)

        dL_dW_conv = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(self.conv_filter_size):
                for j in range(self.conv_filter_size):
                    region = self.X[i:i+self.conv_filter_size, j:j+self.conv_filter_size, :]
                    dL_dW_conv[f, i, j] = np.sum(region * dL_dConv[f])

        dL_db_conv = np.sum(dL_dConv, axis=(1, 2)).reshape(-1, 1)

        # Update weights
        self.W_out -= self.lr * dL_dW_out
        self.b_out -= self.lr * dL_db_out
        self.W_fc -= self.lr * dL_dW_fc
        self.b_fc -= self.lr * dL_db_fc
        self.filters -= self.lr * dL_dW_conv
        self.biases -= self.lr * dL_db_conv

    def train(self, X, y, epochs=10):
        """Train the CNN."""
        for epoch in range(epochs):
            probs = self.forward(X)
            self.backward(y)
            loss = -np.sum(y * np.log(probs))
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
