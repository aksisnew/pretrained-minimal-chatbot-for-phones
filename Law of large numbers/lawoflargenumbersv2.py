import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def initialize_weights(input_dim: int, hidden_dim: int, output_dim: int) -> tuple:
    weights_layer1 = np.random.rand(input_dim, hidden_dim)
    weights_layer2 = np.random.rand(hidden_dim, output_dim)
    return weights_layer1, weights_layer2

def forward_pass(inputs: np.ndarray, weights_layer1: np.ndarray, weights_layer2: np.ndarray) -> tuple:
    layer1 = sigmoid(np.dot(inputs, weights_layer1))
    layer2 = sigmoid(np.dot(layer1, weights_layer2))
    return layer1, layer2

def backward_pass(inputs: np.ndarray, outputs: np.ndarray, layer1: np.ndarray, layer2: np.ndarray, weights_layer1: np.ndarray, weights_layer2: np.ndarray) -> tuple:
    layer2_error = outputs - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(weights_layer2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    return layer1_delta, layer2_delta

def update_weights(inputs: np.ndarray, layer1: np.ndarray, layer1_delta: np.ndarray, layer2_delta: np.dot, learning_rate: float, weights_layer1: np.ndarray, weights_layer2: np.ndarray) -> tuple:
    weights_layer2 += layer1.T.dot(layer2_delta) * learning_rate
    weights_layer1 += inputs.T.dot(layer1_delta) * learning_rate
    return weights_layer1, weights_layer2

def train_network(inputs: np.ndarray, outputs: np.ndarray, hidden_dim: int, num_iterations: int, learning_rate: float) -> tuple:
    weights_layer1, weights_layer2 = initialize_weights(inputs.shape[1], hidden_dim, outputs.shape[1])
    for i in range(num_iterations):
        layer1, layer2 = forward_pass(inputs, weights_layer1, weights_layer2)
        layer1_delta, layer2_delta = backward_pass(inputs, outputs, layer1, layer2, weights_layer1, weights_layer2)
        weights_layer1, weights_layer2 = update_weights(inputs, layer1, layer1_delta, layer2_delta, learning_rate, weights_layer1, weights_layer2)
        if i % 1000 == 0:
            print(f"Error after {i} iterations: {np.mean(np.abs(outputs - layer2))}")
    return weights_layer1, weights_layer2

def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])
    np.random.seed(1)
    hidden_dim = 2
    num_iterations = 20000
    learning_rate = 1
    weights_layer1, weights_layer2 = train_network(inputs, outputs, hidden_dim, num_iterations, learning_rate)
    layer1, layer2 = forward_pass(inputs, weights_layer1, weights_layer2)
    print("Final output:")
    print(layer2)

if __name__ == "__main__":
    main()
