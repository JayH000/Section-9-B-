"""
Copied from Task 1 part b
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class MLP:
    def __init__(self, layers, activation='relu', learning_rate=0.01):
       
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1]) for i in range(1, len(layers))]
        self.biases = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Stability fix
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def softmax_derivative(self, output, Y):
        return output - Y     
       
   

    def forward(self, X):
        
        A = X.T  # Ensure input is column vector
        activations = [A]  
        zs = []  

        for W, b in zip(self.weights, self.biases):
            Z = np.dot(W, A) + b  # Linear transformation
            A = self.softmax(Z) if W.shape[0] == 10 else np.maximum(0, Z)  # Softmax for last layer
            zs.append(Z)
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
            dW[i] =  np.dot(delta, activations[i].T) / m
            db[i] =np.mean(delta, axis=1, keepdims=True)

            # Compute delta for next layer
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * (zs[i-1] > 0)  # ReLU derivative

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
            if epoch % 10 == 0:
               predictions = self.predict(X)
               loss = -np.mean(Y * np.log(predictions + 1e-9))  # Cross-entropy loss
               accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(Y, axis=1))
               losses.append(loss)
               accuracies.append(accuracy)
               print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return losses, accuracies
    
    def predict(self, X):
        """
        Predict class labels.
        """
        return self.forward(X)[0][-1].T  # Return final activations

# Load and preprocess MNIST data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
num_classes = 10

y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# Train MLP
mlp = MLP(layers=[784, 128, 64, 10], learning_rate=0.01)
losses, accuracies = mlp.train(X_train_flat, y_train_onehot, epochs=100)

# Plot convergence
plt.figure()
plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.title("MLP Training Convergence")
plt.show()

# Confusion matrix
predictions = np.argmax(mlp.predict(X_test_flat), axis=1)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot()
plt.title("MLP Confusion Matrix")
plt.show()