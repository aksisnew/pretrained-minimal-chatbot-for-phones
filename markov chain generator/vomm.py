import re
from collections import defaultdict

class VOMM:
    def __init__(self, max_order=5):
        self.max_order = max_order
        self.model = defaultdict(lambda: defaultdict(int))

    def train(self, text):
        tokens = re.findall(r'\w+', text.lower())
        for order in range(1, self.max_order + 1):
            for i in range(len(tokens) - order):
                context = tuple(tokens[i:i + order])
                next_token = tokens[i + order]
                self.model[context][next_token] += 1

    def generate(self, context, length=100):
        output = list(context)
        for _ in range(length):
            context = tuple(output[-self.max_order:])
            next_tokens = self.model[context]
            if not next_tokens:
                break
            next_token = max(next_tokens, key=next_tokens.get)
            output.append(next_token)
        return ' '.join(output)

# Load the text corpus
with open('corpus.txt', 'r') as f:
    text = f.read()

# Train the VOMM model
vomm = VOMM()
vomm.train(text)

# Generate text
context = ['the', 'quick', 'brown']
print(vomm.generate(context))
