import re

class LZ77TextGenerator:
    def __init__(self, corpus_file):
            self.corpus = open(corpus_file, 'r').read()
                    self.corpus = re.sub(r'\s+', ' ', self.corpus).lower()
                            self.dictionary = {}

                                def compress(self):
                                        i = 0
                                                while i < len(self.corpus):
                                                            for j in range(i + 1, len(self.corpus) + 1):
                                                                            substring = self.corpus[i:j]
                                                                                            if substring not in self.dictionary:
                                                                                                                self.dictionary[substring] = len(self.dictionary)
                                                                                                                                    yield (self.corpus[i-1] if i > 0 else '', substring)
                                                                                                                                                        i = j
                                                                                                                                                                            break

                                                                                                                                                                                def generate_text(self, length):
                                                                                                                                                                                        generated_text = ''
                                                                                                                                                                                                compressor = self.compress()
                                                                                                                                                                                                        while len(generated_text) < length:
                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                    prefix, substring = next(compressor)
                                                                                                                                                                                                                                                    generated_text += substring
                                                                                                                                                                                                                                                                except StopIteration:
                                                                                                                                                                                                                                                                                break
                                                                                                                                                                                                                                                                                        return generated_text[:length]

                                                                                                                                                                                                                                                                                        # Usage
                                                                                                                                                                                                                                                                                        generator = LZ77TextGenerator('corpus.txt')
                                                                                                                                                                                                                                                                                        print(generator.generate_text(100))