import re
from hmmlearn import hmm
import numpy as np

# Load the text file
with open('corpus.txt', 'r') as f:
    text = f.read()

# Preprocess the text
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
words = text.split()

# Create a dictionary to map words to integers
word_dict = {}
for word in words:
    if word not in word_dict:
        word_dict[word] = len(word_dict)

# Convert the text to a sequence of integers
seq = [word_dict[word] for word in words]

# Create an HMM model
model = hmm.MultinomialHMM(n_components=5)

# Fit the model to the data
seq = np.array(seq).reshape(-1, 1)
model.fit(seq)

# Generate text
def generate_text(model, length):
    state = model.predict(seq[-1].reshape(1, -1))[0]
    output = []
    for _ in range(length):
        output.append(list(word_dict.keys())[list(word_dict.values()).index(np.random.choice(model.n_features, p=model.emissionprob_[state]))])
        state = np.random.choice(model.n_components, p=model.transmat_[state])
    return ' '.join(output)

print(generate_text(model, 100))
