import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.power(x, 2)

class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 activation='sigmoid', learning_rate=0.1, momentum=0.9):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights_layer1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.weights_layer2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_layer1 = np.zeros((1, hidden_dim))
        self.bias_layer2 = np.zeros((1, output_dim))

        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Unsupported activation function")

        # Momentum terms
        self.v_w1 = np.zeros_like(self.weights_layer1)
        self.v_w2 = np.zeros_like(self.weights_layer2)
        self.v_b1 = np.zeros_like(self.bias_layer1)
        self.v_b2 = np.zeros_like(self.bias_layer2)

    def forward(self, X: np.ndarray) -> tuple:
        self.layer1 = self.activation(np.dot(X, self.weights_layer1) + self.bias_layer1)
        self.layer2 = self.activation(np.dot(self.layer1, self.weights_layer2) + self.bias_layer2)
        return self.layer2

    def backward(self, X: np.ndarray, y: np.ndarray):
        layer2_error = y - self.layer2
        layer2_delta = layer2_error * self.activation_derivative(self.layer2)

        layer1_error = layer2_delta.dot(self.weights_layer2.T)
        layer1_delta = layer1_error * self.activation_derivative(self.layer1)

        return layer1_delta, layer2_delta

    def update_weights(self, X: np.ndarray, layer1_delta: np.ndarray, layer2_delta: np.ndarray):
        # Momentum update
        self.v_w2 = self.momentum * self.v_w2 + self.learning_rate * self.layer1.T.dot(layer2_delta)
        self.v_w1 = self.momentum * self.v_w1 + self.learning_rate * X.T.dot(layer1_delta)
        self.v_b2 = self.momentum * self.v_b2 + self.learning_rate * np.sum(layer2_delta, axis=0, keepdims=True)
        self.v_b1 = self.momentum * self.v_b1 + self.learning_rate * np.sum(layer1_delta, axis=0, keepdims=True)

        self.weights_layer2 += self.v_w2
        self.weights_layer1 += self.v_w1
        self.bias_layer2 += self.v_b2
        self.bias_layer1 += self.v_b1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, verbose: bool = True):
        history = []
        for i in tqdm(range(epochs), disable=not verbose):
            self.forward(X)
            layer1_delta, layer2_delta = self.backward(X, y)
            self.update_weights(X, layer1_delta, layer2_delta)
            if verbose and i % 1000 == 0:
                loss = np.mean(np.abs(y - self.layer2))
                history.append(loss)
                print(f"Epoch {i}, Loss: {loss}")
        return history

# Example usage
def main():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, 
                       activation='tanh', learning_rate=0.1, momentum=0.8)
    nn.train(X, y, epochs=10000)
    output = nn.forward(X)
    print("Final output:")
    print(np.round(output, 3))

if __name__ == "__main__":
    main()
