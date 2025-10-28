import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define constants for categories and keywords
NAME_KEYWORDS = ["name", "person", "individual"]
PLACE_KEYWORDS = ["place", "location", "area"]
PEOPLE_KEYWORDS = ["people", "crowd", "group"]
THINGS_KEYWORDS = ["thing", "object", "item"]
MOOD_KEYWORDS = ["happy", "sad", "angry", "excited"]
GREETING_KEYWORDS = ["hello", "hi", "hey", "greetings"]
FAREWELL_KEYWORDS = ["bye", "goodbye", "see you", "farewell"]

CATEGORIES = {
    "NAME": NAME_KEYWORDS,
    "PLACE": PLACE_KEYWORDS,
    "PEOPLE": PEOPLE_KEYWORDS,
    "THINGS": THINGS_KEYWORDS,
    "MOOD": MOOD_KEYWORDS,
    "GREETING": GREETING_KEYWORDS,
    "FAREWELL": FAREWELL_KEYWORDS
}

# Define a function to categorize tokens
def categorize_tokens(doc):
    categories = []
    for token in doc:
        for category, keywords in CATEGORIES.items():
            if token.text.lower() in [keyword.lower() for keyword in keywords]:
                categories.append(category)
    return categories

# Define a function to process input
def process_input(input_text):
    doc = nlp(input_text)
    categories1 = categorize_tokens(doc)
    categories2 = categorize_tokens(doc)
    
    return {
        "input_text": input_text,
        "categories": list(set(categories1 + categories2))
    }

# Test the function
input_text = input("Enter a sentence: ")
result = process_input(input_text)

print("Input Text:", result["input_text"])
print("Categories:", result["categories"])
