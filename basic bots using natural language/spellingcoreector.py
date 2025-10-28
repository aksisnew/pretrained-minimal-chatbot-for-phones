import re
from collections import Counter
import numpy as np

class SpellingCorrector:
    def __init__(self, text, lowercase=True):
        # Convert text to lowercase if desired
        self.text = text.lower() if lowercase else text
        
        # Extract words (letters only)
        self.words = re.findall(r'\b[a-z]+\b', self.text)
        
        # Count word frequencies
        self.word_counts = Counter(self.words)
        
        # Total word count
        self.total_words = sum(self.word_counts.values())
        
        # Laplace smoothing factor
        self.smoothing = 1

    def probability(self, word):
        # Add smoothing to avoid zero probability
        return (self.word_counts[word] + self.smoothing) / (self.total_words + self.smoothing * len(self.word_counts))

    def edit_distance(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return dp[m, n]

    def known(self, words):
        return set(w for w in words if w in self.word_counts)

    def correct(self, word):
        candidates = (
            self.known([word]) or 
            self.known(self.edits1(word)) or 
            self.known(self.edits2(word)) or 
            [word]
        )
        return max(candidates, key=self.probability)

    def edits1(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


def main():
    # Load your corpus
    with open('corpus.txt', 'r') as f:
        text = f.read()
    
    corrector = SpellingCorrector(text)
    
    print("Welcome to the Spelling Corrector! Type 'quit' to exit.")
    while True:
        word = input("Enter a word: ").strip()
        if word.lower() == 'quit':
            print("Goodbye! Keep spelling well! 👋")
            break
        
        corrected_word = corrector.correct(word.lower())
        if word.lower() != corrected_word:
            print(f"Did you mean '{corrected_word}'? 🤔")
        else:
            print("Word is spelled correctly! ✅")


if __name__ == "__main__":
    main()
