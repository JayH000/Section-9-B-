import numpy as np

class MLP:
    def __init__(self, layers, activation='relu', learning_rate=0.01):
        """
        layers: List specifying number of neurons per layer.
        activation: 'relu' or 'sigmoid'.
        learning_rate: Step size for weight updates.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation_func = self.relu if activation == 'relu' else self.sigmoid
        self.activation_derivative = self.relu_derivative if activation == 'relu' else self.sigmoid_derivative
        
        # Initialize weights and biases
        self.weights = [np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1]) for i in range(1, len(layers))]
        self.biases = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X):
        """
        Perform forward pass through the network.
        """
        A = X.T  # Ensure input is column vector
        activations = [A]  
        zs = []  

        for W, b in zip(self.weights, self.biases):
            Z = np.dot(W, A) + b  # Linear transformation
            A = self.activation_func(Z)  # Apply activation
            zs.append(Z)
            activations.append(A)

        return activations, zs

    def backward(self, X, Y):
        """
        Perform backward propagation and update weights.
        """
        m = X.shape[0]  # Number of samples
        activations, zs = self.forward(X)
        
        # Initialize gradient storage
        dW, db = [None] * len(self.weights), [None] * len(self.biases)

        # Compute output layer error
        delta = (activations[-1] - Y.T) * self.activation_derivative(zs[-1])

        # Backpropagate errors
        for i in reversed(range(len(self.weights))):
            dW[i] = np.einsum("bi,bj->bij", delta.T, activations[i].T).mean(axis=0)
            db[i] = np.mean(delta, axis=1, keepdims=True)

            # Compute delta for next layer
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.activation_derivative(zs[i-1])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, X, Y, epochs=1000):
        """
        Train the network using gradient descent.
        """
        for epoch in range(epochs):
            self.backward(X, Y)
            if epoch % 100 == 0:
                loss = np.mean((self.forward(X)[0][-1] - Y.T) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predict class labels.
        """
        return self.forward(X)[0][-1].T  # Return final activations

# Example Usage
if __name__ == "__main__":
    # Example: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    mlp = MLP(layers=[2, 4, 2, 1], activation='relu', learning_rate=0.1)
    mlp.train(X, Y, epochs=1000)

    print("Predictions:", mlp.predict(X))
