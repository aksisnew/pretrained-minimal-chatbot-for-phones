# advanced_pos_tagger.py

import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from textblob import TextBlob
from collections import Counter
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')

# =========================
# SpaCy-based POS Tagger
# =========================
# Load large English model (more accurate than small)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

def spacy_pos_tagger(text: str):
    """
    Uses SpaCy to tokenize and POS-tag a text
    """
    doc = nlp(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]

# =========================
# NLTK POS Tagger
# =========================
def nltk_pos_tagger(text: str):
    """
    Tokenizes sentences and words, then tags with NLTK's default tagger
    """
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)

# =========================
# Trainable NLTK Tagger with Treebank corpus
# =========================
def train_ngram_tagger():
    """
    Trains a combined unigram/bigram/trigram tagger on the NLTK Treebank corpus
    """
    sentences = treebank.tagged_sents()
    unigram_tagger = UnigramTagger(sentences)
    bigram_tagger = BigramTagger(sentences, backoff=unigram_tagger)
    trigram_tagger = TrigramTagger(sentences, backoff=bigram_tagger)
    return trigram_tagger

trained_tagger = train_ngram_tagger()

def custom_nltk_tagger(text: str):
    tokens = word_tokenize(text)
    return trained_tagger.tag(tokens)

# =========================
# TextBlob POS Tagger
# =========================
def textblob_pos_tagger(text: str):
    blob = TextBlob(text)
    return blob.tags

# =========================
# Utility: Count POS occurrences
# =========================
def pos_frequency(tagged_tokens):
    counter = Counter(tag for _, tag in tagged_tokens)
    return counter

# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming the way we live and work.
    Natural language processing is a core component of AI applications.
    """

    print("=== SpaCy POS Tagger ===")
    print(spacy_pos_tagger(sample_text))

    print("\n=== NLTK Default POS Tagger ===")
    print(nltk_pos_tagger(sample_text))

    print("\n=== Custom Trigram NLTK Tagger ===")
    print(custom_nltk_tagger(sample_text))

    print("\n=== TextBlob POS Tagger ===")
    print(textblob_pos_tagger(sample_text))
