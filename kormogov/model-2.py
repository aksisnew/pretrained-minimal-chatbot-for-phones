import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from scipy.stats import entropy

# Download NLTK data
nltk.download('punkt', quiet=True)

class AdvancedKolmogorovMarkovGenerator:
    def __init__(self, corpus_file, order=2, tokenize_by_word=False):
        self.order = order
        self.tokenize_by_word = tokenize_by_word
        
        # Load and clean corpus
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.corpus = f.read().lower()
        self.corpus = re.sub(r'\s+', ' ', self.corpus)
        
        # Tokenize
        if tokenize_by_word:
            self.tokens = word_tokenize(self.corpus)
        else:
            self.tokens = list(self.corpus)
        
        # Build n-grams
        self.ngrams_list = list(ngrams(self.tokens, self.order + 1))
        
        # Build Markov chain
        self.markov_chain = self.build_markov_chain()
        
        # Build substring dictionary for Kolmogorov score
        self.dictionary = self.build_dictionary()
    
    def build_markov_chain(self):
        """Create Markov chain with probability distributions"""
        chain = defaultdict(Counter)
        for ngram in self.ngrams_list:
            context = ngram[:-1]
            next_token = ngram[-1]
            chain[context][next_token] += 1
        # Normalize counts to probabilities
        for context, counter in chain.items():
            total = sum(counter.values())
            for token in counter:
                counter[token] /= total
        return chain
    
    def build_dictionary(self):
        """Dictionary of substring lengths for Kolmogorov score"""
        dictionary = {}
        joined_tokens = ''.join(self.tokens)
        for i in range(len(joined_tokens)):
            for j in range(i + 1, len(joined_tokens) + 1):
                substring = joined_tokens[i:j]
                if substring not in dictionary:
                    dictionary[substring] = len(substring)
        return dictionary
    
    def kolmogorov_score(self, substring):
        """Simplified Kolmogorov complexity approximation"""
        return len(substring) / (1 + self.dictionary.get(substring, 0))
    
    def generate_text(self, length=100):
        """Generate text using Markov chain and Kolmogorov scoring"""
        context = random.choice(list(self.markov_chain.keys()))
        generated_tokens = list(context)
        
        while len(generated_tokens) < length:
            next_probs = self.markov_chain.get(tuple(generated_tokens[-self.order:]), None)
            if not next_probs:
                break
            # Weighted random selection using probabilities and Kolmogorov score
            tokens = list(next_probs.keys())
            probs = np.array(list(next_probs.values()))
            kolmogorov_weights = np.array([self.kolmogorov_score(''.join(generated_tokens[-self.order:] + (t,))) for t in tokens])
            combined_weights = probs * kolmogorov_weights
            combined_weights /= combined_weights.sum()
            next_token = np.random.choice(tokens, p=combined_weights)
            generated_tokens.append(next_token)
        
        if self.tokenize_by_word:
            return ' '.join(generated_tokens[:length])
        else:
            return ''.join(generated_tokens[:length])
    
    def most_common_ngrams(self, top_n=10):
        """Return most common n-grams in corpus"""
        flat_ngrams = [''.join(ng) if not self.tokenize_by_word else ' '.join(ng) for ng in self.ngrams_list]
        return pd.Series(flat_ngrams).value_counts().head(top_n)

    def entropy_of_corpus(self):
        """Compute Shannon entropy of token distribution"""
        freq = Counter(self.tokens)
        probs = np.array(list(freq.values())) / sum(freq.values())
        return entropy(probs, base=2)

# Usage Example
if __name__ == "__main__":
    generator = AdvancedKolmogorovMarkovGenerator('corpus.txt', order=3, tokenize_by_word=True)
    
    print("=== Generated Text ===")
    print(generator.generate_text(50))
    
    print("\n=== Most Common N-grams ===")
    print(generator.most_common_ngrams(5))
    
    print("\n=== Entropy of Corpus ===")
    print(generator.entropy_of_corpus())
