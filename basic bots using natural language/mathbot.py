import nltk
from nltk.tokenize import word_tokenize
import math
import re

# Initialize NLTK
nltk.download('punkt')

# Define a dictionary of math functions
math_functions = {
    'sin': math.sin,
        'cos': math.cos,
            'tan': math.tan,
                'asin': math.asin,
                    'acos': math.acos,
                        'atan': math.atan,
                            'sqrt': math.sqrt,
                                'log': math.log,
                                    'exp': math.exp,
                                        'abs': abs,
                                            'ceil': math.ceil,
                                                'floor': math.floor,
                                                }

                                                # Define a dictionary of geometry functions
                                                geometry_functions = {
                                                    'area of circle': lambda r: math.pi * r ** 2,
                                                        'circumference of circle': lambda r: 2 * math.pi * r,
                                                            'area of rectangle': lambda l, w: l * w,
                                                                'perimeter of rectangle': lambda l, w: 2 * (l + w),
                                                                    'area of triangle': lambda b, h: 0.5 * b * h,
                                                                        'perimeter of triangle': lambda a, b, c: a + b + c,
                                                                        }

                                                                        def process_input(input_str):
                                                                            # Tokenize the input string
                                                                                tokens = word_tokenize(input_str)

                                                                                    # Check if the input string is a math expression
                                                                                        if any(token in math_functions for token in tokens):
                                                                                                # Extract the math function and arguments
                                                                                                        func_name = [token for token in tokens if token in math_functions][0]
                                                                                                                args = [float(token) for token in tokens if token.replace('.', '', 1).isdigit()]
                                                                                                                        # Evaluate the math expression
                                                                                                                                result = math_functions[func_name](*args)
                                                                                                                                        return f"The result is {result}"

                                                                                                                                            # Check if the input string is a geometry problem
                                                                                                                                                for func_name, func in geometry_functions.items():
                                                                                                                                                        if func_name in input_str:
                                                                                                                                                                    # Extract the arguments
                                                                                                                                                                                args = re.findall(r'\d+\.\d+|\d+', input_str)
                                                                                                                                                                                            args = [float(arg) for arg in args]
                                                                                                                                                                                                        # Evaluate the geometry problem
                                                                                                                                                                                                                    result = func(*args)
                                                                                                                                                                                                                                return f"The result is {result}"

                                                                                                                                                                                                                                    # If the input string is not recognized, return an error message
                                                                                                                                                                                                                                        return "Sorry, I didn't understand that."

                                                                                                                                                                                                                                        def main():
                                                                                                                                                                                                                                            while True:
                                                                                                                                                                                                                                                    user_input = input("Enter a math problem or type 'quit' to exit: ")
                                                                                                                                                                                                                                                            if user_input.lower() == 'quit':
                                                                                                                                                                                                                                                                        break
                                                                                                                                                                                                                                                                                print(process_input(user_input))

                                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                                    main()