import numpy as np
import struct

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number in image file!")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0  # Normalize to [0,1]  

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in label file!")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels



# File paths
train_images_path = "/Users/domholguin/Documents/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte"
train_labels_path = "/Users/domholguin/Documents/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
test_images_path = "/Users/domholguin/Documents/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
test_labels_path = "/Users/domholguin/Documents/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"




# Load dataset
try:
    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    raise

# Flatten images for MLP
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# One-hot encoding for labels
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

print("Data loaded and preprocessed successfully!")
