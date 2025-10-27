import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function
    def sigmoid_derivative(x):
        return x * (1 - x)

        # Input datasets
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Output datasets
        y = np.array([[0], [1], [1], [0]])

        # Seed random numbers to make calculation deterministic
        np.random.seed(1)

        # Initialize weights randomly with mean 0
        weight0 = 2 * np.random.random((2, 2)) - 1
        weight1 = 2 * np.random.random((2, 1)) - 1

        # Number of iterations
        num_iterations = 20000

        # Learning rate
        learning_rate = 1

        # Training loop
        for i in range(num_iterations):
            # Forward pass
                layer0 = X
                    layer1 = sigmoid(np.dot(layer0, weight0))
                        layer2 = sigmoid(np.dot(layer1, weight1))

                            # Backward pass
                                layer2_error = y - layer2
                                    layer2_delta = layer2_error * sigmoid_derivative(layer2)
                                        layer1_error = layer2_delta.dot(weight1.T)
                                            layer1_delta = layer1_error * sigmoid_derivative(layer1)

                                                # Weight updates
                                                    weight1 += layer1.T.dot(layer2_delta) * learning_rate
                                                        weight0 += layer0.T.dot(layer1_delta) * learning_rate

                                                            # Print error every 1000 iterations
                                                                if i % 1000 == 0:
                                                                        print("Error after {} iterations: {}".format(i, np.mean(np.abs(layer2_error))))

                                                                        # Print final output
                                                                        print("Final output:")
                                                                        print(layer2)