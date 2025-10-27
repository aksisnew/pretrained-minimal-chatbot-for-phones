import re
from collections import defaultdict
import random
import time
from sklearn.metrics import perplexity
from nltk.util import ngrams

class MarkovNGramBenchmark:
    def __init__(self, corpus_file, n=8):
        self.corpus = open(corpus_file, 'r').read()
        self.corpus = re.sub(r'\s+', ' ', self.corpus).lower()
        self.n = n
        self.markov_chain = self.build_markov_chain()

    def build_markov_chain(self):
        markov_chain = defaultdict(list)
        n_grams = ngrams(self.corpus, self.n + 1)
        for gram in n_grams:
            context = gram[:-1]
            next_char = gram[-1]
            markov_chain[context].append(next_char)
        return markov_chain

    def generate_text(self, length):
        context = random.choice(list(self.markov_chain.keys()))
        generated_text = ''.join(context)
        for _ in range(length - self.n):
            next_chars = self.markov_chain[context]
            next_char = random.choice(next_chars)
            generated_text += next_char
            context = context[1:] + (next_char,)
        return generated_text

    def evaluate_perplexity(self, test_corpus):
        test_ngrams = ngrams(test_corpus, self.n + 1)
        log_likelihood = 0
        total_ngrams = 0
        for gram in test_ngrams:
            context = gram[:-1]
            next_char = gram[-1]
            if context in self.markov_chain:
                next_chars = self.markov_chain[context]
                prob = 1 / len(next_chars) if next_char in next_chars else 0
                log_likelihood += np.log2(prob) if prob > 0 else -np.inf
                total_ngrams += 1
        perplexity = 2 ** (-log_likelihood / total_ngrams)
        return perplexity

    def benchmark(self, test_corpus):
        start_time = time.time()
        generated_text = self.generate_text(len(test_corpus))
        end_time = time.time()
        perplexity_score = self.evaluate_perplexity(test_corpus)
        metrics = {
            'Perplexity': perplexity_score,
            'Generation Time': end_time - start_time,
            'Generated Text Length': len(generated_text)
        }
        return metrics

# Usage
benchmark = MarkovNGramBenchmark('corpus.txt')
test_corpus = open('test_corpus.txt', 'r').read()
test_corpus = re.sub(r'\s+', ' ', test_corpus).lower()
metrics = benchmark.benchmark(test_corpus)
for metric, value in metrics.items():
    print(f'{metric}: {value}')
