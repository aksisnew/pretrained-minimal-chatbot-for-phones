import numpy as np

class BernoulliNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_counts = {}
        self.feature_counts = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_counts[c] = len(X_c)
            self.feature_counts[c] = np.sum(X_c, axis=0)

        self.total_samples = len(X)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_counts[c] / self.total_samples)
                posterior = prior
                for i in range(len(x)):
                    if x[i] == 1:
                        posterior += np.log((self.feature_counts[c][i] + 1) / (self.class_counts[c] + 2))
                    else:
                        posterior += np.log(1 - (self.feature_counts[c][i] + 1) / (self.class_counts[c] + 2))
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

# Generate some random data
np.random.seed(0)
X = np.random.randint(0, 2, (100, 10))
y = np.random.randint(0, 2, 100)

# Train the model
bnb = BernoulliNaiveBayes()
bnb.fit(X, y)

# Make predictions
predictions = bnb.predict(X)

# Evaluate the model
accuracy = np.sum(predictions == y) / len(y)
print('Accuracy:', accuracy)
