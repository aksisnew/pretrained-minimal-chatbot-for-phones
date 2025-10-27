import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.variance = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(len([i for i in self.classes if i == c]) / len(self.classes))
                posterior = prior
                for i in range(len(x)):
                    posterior += np.log(self._pdf(x[i], self.mean[c][i], self.variance[c][i]))
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

    def _pdf(self, x, mean, variance):
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Train the model
gnb = GaussianNaiveBayes()
gnb.fit(X, y)

# Make predictions
predictions = gnb.predict(X)

# Evaluate the model
accuracy = np.sum(predictions == y) / len(y)
print('Accuracy:', accuracy)
