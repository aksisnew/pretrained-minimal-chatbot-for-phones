import re
from collections import defaultdict
import random

class KolmogorovMarkovGenerator:
    def __init__(self, corpus_file, order=2):
        self.corpus = open(corpus_file, 'r').read()
        self.corpus = re.sub(r'\s+', ' ', self.corpus).lower()
        self.order = order
        self.markov_chain = self.build_markov_chain()
        self.dictionary = self.build_dictionary()

    def build_markov_chain(self):
        markov_chain = defaultdict(list)
        for i in range(len(self.corpus) - self.order):
            context = self.corpus[i:i + self.order]
            next_char = self.corpus[i + self.order]
            markov_chain[context].append(next_char)
        return markov_chain

    def build_dictionary(self):
        dictionary = {}
        for i in range(len(self.corpus)):
            for j in range(i + 1, len(self.corpus) + 1):
                substring = self.corpus[i:j]
                if substring not in dictionary:
                    dictionary[substring] = len(substring)
        return dictionary

    def kolmogorov_score(self, substring):
        return len(substring) / (1 + self.dictionary.get(substring, 0))

    def generate_text(self, length):
        context = random.choice(list(self.markov_chain.keys()))
        generated_text = context
        while len(generated_text) < length:
            next_chars = self.markov_chain[context]
            next_char = max(next_chars, key=lambda x: self.kolmogorov_score(context + x))
            generated_text += next_char
            context = generated_text[-self.order:]
        return generated_text[:length]

# Usage
generator = KolmogorovMarkovGenerator('corpus.txt')
print(generator.generate_text(100))
