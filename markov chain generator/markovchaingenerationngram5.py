import re
from collections import defaultdict
import random

class MarkovChain:
    def __init__(self, n=5):
        self.n = n
        self.model = defaultdict(list)

    def train(self, text):
        tokens = re.findall(r'\w+', text.lower())
        for i in range(len(tokens) - self.n):
            context = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]
            self.model[context].append(next_token)

    def generate(self, context, max_length=100):
        sentence = list(context)
        for _ in range(max_length):
            next_tokens = self.model[tuple(sentence[-self.n:])]
            if not next_tokens:
                break
            next_token = random.choice(next_tokens)
            sentence.append(next_token)
        return ' '.join(sentence)

def load_corpus(filename):
    with open(filename, 'r') as f:
        return f.read()

def main():
    corpus = load_corpus('corpus.txt')
    generator = MarkovChain(n=5)
    generator.train(corpus)

    context = ['the', 'quick', 'brown', 'fox', 'jumps']
    print(generator.generate(context))

if __name__ == '__main__':
    main()
