import numpy as np

class CategoricalNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_counts = {}
        self.feature_counts = {}
        self.feature_values = {}

        for i in range(X.shape[1]):
            self.feature_values[i] = np.unique(X[:, i])

        for c in self.classes:
            X_c = X[y == c]
            self.class_counts[c] = len(X_c)
            self.feature_counts[c] = {}

            for i in range(X.shape[1]):
                self.feature_counts[c][i] = {}
                for value in self.feature_values[i]:
                    self.feature_counts[c][i][value] = np.sum(X_c[:, i] == value)

        self.total_samples = len(X)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_counts[c] / self.total_samples)
                posterior = prior
                for i in range(len(x)):
                    posterior += np.log((self.feature_counts[c][i].get(x[i], 0) + 1) / (self.class_counts[c] + len(self.feature_values[i])))
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

# Generate some random data
np.random.seed(0)
X = np.random.randint(0, 3, (100, 5))
y = np.random.randint(0, 2, 100)

# Train the model
cnb = CategoricalNaiveBayes()
cnb.fit(X, y)

# Make predictions
predictions = cnb.predict(X)

# Evaluate the model
accuracy = np.sum(predictions == y) / len(y)
print('Accuracy:', accuracy)
