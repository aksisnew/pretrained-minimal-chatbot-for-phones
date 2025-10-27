import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Pre-fed dictionary
dictionary = {
    'positive': ['good', 'great', 'excellent', 'amazing', 'awesome'],
        'negative': ['bad', 'terrible', 'awful', 'poor', 'horrible'],
            'neutral': ['okay', 'fine', 'alright', 'decent', 'average']
            }

            # Training data
            train_data = [
                ('I love this product!', 'positive'),
                    ('This product is terrible.', 'negative'),
                        ('It\'s okay, I guess.', 'neutral'),
                            ('This product is amazing!', 'positive'),
                                ('I hate this product.', 'negative'),
                                    ('It\'s decent, I suppose.', 'neutral'),
                                        ('I\'m so happy with this product!', 'positive'),
                                            ('This product is awful.', 'negative'),
                                                ('It\'s fine, I guess.', 'neutral')
                                                ]

                                                # Split data into input text and labels
                                                text_data = [item[0] for item in train_data]
                                                labels = [item[1] for item in train_data]

                                                # Create a CountVectorizer object
                                                vectorizer = CountVectorizer()

                                                # Fit the vectorizer to the text data and transform it into a matrix of token counts
                                                X = vectorizer.fit_transform(text_data)

                                                # Split the data into training and testing sets
                                                X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

                                                class NaiveBayesClassifier:
                                                    def __init__(self):
                                                            self.classes = set()
                                                                    self.log_prior = {}
                                                                            self.log_likelihood = {}

                                                                                def fit(self, X, y):
                                                                                        self.classes = set(y)
                                                                                                n_classes = len(self.classes)
                                                                                                        n_features = X.shape[1]

                                                                                                                # Calculate the log prior probabilities
                                                                                                                        for c in self.classes:
                                                                                                                                    self.log_prior[c] = np.log(len([label for label in y if label == c]) / len(y))

                                                                                                                                            # Calculate the log likelihood probabilities
                                                                                                                                                    for c in self.classes:
                                                                                                                                                                X_c = X[np.array(y) == c]
                                                                                                                                                                            self.log_likelihood[c] = np.log((X_c.sum(axis=0) + 1) / (X_c.sum() + n_features))

                                                                                                                                                                                def predict(self, X):
                                                                                                                                                                                        predictions = []
                                                                                                                                                                                                for x in X:
                                                                                                                                                                                                            posterior_probabilities = []
                                                                                                                                                                                                                        for c in self.classes:
                                                                                                                                                                                                                                        posterior_probability = self.log_prior[c]
                                                                                                                                                                                                                                                        posterior_probability += np.sum(x * self.log_likelihood[c])
                                                                                                                                                                                                                                                                        posterior_probabilities.append(posterior_probability)
                                                                                                                                                                                                                                                                                    predictions.append(list(self.classes)[np.argmax(posterior_probabilities)])
                                                                                                                                                                                                                                                                                            return predictions

                                                                                                                                                                                                                                                                                            # Train the Naive Bayes classifier
                                                                                                                                                                                                                                                                                            clf = NaiveBayesClassifier()
                                                                                                                                                                                                                                                                                            clf.fit(X_train, y_train)

                                                                                                                                                                                                                                                                                            # Make predictions on the test set
                                                                                                                                                                                                                                                                                            y_pred = clf.predict(X_test)

                                                                                                                                                                                                                                                                                            # Evaluate the classifier
                                                                                                                                                                                                                                                                                            print('Accuracy:', accuracy_score(y_test, y_pred))
                                                                                                                                                                                                                                                                                            print('Classification Report:')
                                                                                                                                                                                                                                                                                            print(classification_report(y_test, y_pred))
                                                                                                                                                                                                                                                                                            print('Confusion Matrix:')
                                                                                                                                                                                                                                                                                            print(confusion_matrix(y_test, y_pred))

                                                                                                                                                                                                                                                                                            # Test the classifier with a new text
                                                                                                                                                                                                                                                                                            new_text = ['I love this product!']
                                                                                                                                                                                                                                                                                            new_text_vector = vectorizer.transform(new_text)
                                                                                                                                                                                                                                                                                            prediction = clf.predict(new_text_vector)
                                                                                                                                                                                                                                                                                            print('Prediction:', prediction[0])