import nltk
from nltk.util import ngrams
from collections import Counter
import re

# Load the text file
with open('data.txt', 'r') as f:
    text = f.read()

    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Create a 2-gram model
    bigrams = ngrams(tokens, 2)

    # Count the frequency of each bigram
    bigram_counts = Counter(bigrams)

    # Print the top 10 most common bigrams
    print(bigram_counts.most_common(10))

    # Create a function to generate text based on the bigram model
    def generate_text(bigram_counts, start_word, num_words):
        sentence = [start_word]
            for i in range(num_words):
                    next_words = [word for (word1, word) in bigram_counts if word1 == sentence[-1]]
                            next_word_counts = Counter({word: bigram_counts[(sentence[-1], word)] for word in next_words})
                                    next_word = next_word_counts.most_common(1)[0][0]
                                            sentence.append(next_word)
                                                return ' '.join(sentence)

                                                # Generate text based on the bigram model
                                                print(generate_text(bigram_counts, 'the', 10))