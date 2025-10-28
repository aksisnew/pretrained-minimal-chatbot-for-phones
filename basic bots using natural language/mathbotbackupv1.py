import nltk
from nltk.tokenize import word_tokenize
import math
import re
import ast
import operator as op

# Initialize NLTK
nltk.download('punkt', quiet=True)

# Safe arithmetic evaluation
operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg
}

def safe_eval(expr):
    """Safely evaluate basic arithmetic expressions."""
    try:
        def _eval(node):
            if isinstance(node, ast.Num):  # number
                return node.n
            elif isinstance(node, ast.BinOp):  # binary op
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):  # unary op
                return operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(node)
        return _eval(ast.parse(expr, mode='eval').body)
    except Exception:
        return None

# Helper: convert trig input from degrees to radians
def trig_in_degrees(func, x):
    radians = math.radians(x)
    return func(radians)

# Define math functions
math_functions = {
    # --- Trigonometric functions (degrees) ---
    'sin': lambda x: trig_in_degrees(math.sin, x),
    'cos': lambda x: trig_in_degrees(math.cos, x),
    'tan': lambda x: trig_in_degrees(math.tan, x),
    'asin': lambda x: math.degrees(math.asin(x)),
    'acos': lambda x: math.degrees(math.acos(x)),
    'atan': lambda x: math.degrees(math.atan(x)),

    # --- Reciprocal trig functions ---
    'csc': lambda x: 1 / trig_in_degrees(math.sin, x),
    'sec': lambda x: 1 / trig_in_degrees(math.cos, x),
    'cot': lambda x: 1 / trig_in_degrees(math.tan, x),

    # --- Hyperbolic functions ---
    'sinh': math.sinh,
    'cosh': math.cosh,
    'tanh': math.tanh,

    # --- Inverse hyperbolic functions ---
    'asinh': math.asinh,
    'acosh': math.acosh,
    'atanh': math.atanh,

    # --- Basic operations ---
    'sqrt': math.sqrt,
    'cbrt': lambda x: x ** (1/3),
    'square': lambda x: x ** 2,
    'cube': lambda x: x ** 3,
    'pow': pow,
    'log': math.log,       # natural log
    'log10': math.log10,   # base 10
    'exp': math.exp,
    'abs': abs,
    'ceil': math.ceil,
    'floor': math.floor,
    'round': round,
    'mod': lambda a, b: a % b,

    # --- Combinatorics ---
    'factorial': math.factorial,
    'perm': math.perm,   # permutations
    'comb': math.comb,   # combinations

# --- added cube root ---
'max': lambda *args: max(args),
'min': lambda *args: min(args),
'sum': lambda *args: sum(args),


    # --- Other utilities ---
    'degrees': math.degrees,
    'radians': math.radians,
    'sign': lambda x: (x > 0) - (x < 0),
    'hypot': math.hypot   # √(x² + y²)
}

# Define constants
constants = {
    'pi': math.pi,
    'e': math.e,
    'phi': (1 + math.sqrt(5)) / 2
}

def replace_constants(expr):
    """Replace known constants (pi, e, phi) in a string."""
    for name, val in constants.items():
        expr = expr.replace(name, str(val))
    return expr

# Define geometry functions
geometry_functions = {
    'area of circle': lambda r: math.pi * r ** 2,
    'circumference of circle': lambda r: 2 * math.pi * r,
    'area of rectangle': lambda l, w: l * w,
    'perimeter of rectangle': lambda l, w: 2 * (l + w),
    'area of triangle': lambda b, h: 0.5 * b * h,
    'perimeter of triangle': lambda a, b, c: a + b + c,
    'volume of sphere': lambda r: (4 / 3) * math.pi * r ** 3,
    'surface area of sphere': lambda r: 4 * math.pi * r ** 2,
    'volume of cylinder': lambda r, h: math.pi * r ** 2 * h,
    'surface area of cylinder': lambda r, h: 2 * math.pi * r * (r + h),
    'pythagoras': lambda a, b: math.sqrt(a ** 2 + b ** 2)
}

def process_input(input_str):
    # tokenizing taken input
    tokens = word_tokenize(input_str.lower())
    input_str = input_str.lower()

    # Replace constants
    input_str = replace_constants(input_str)

    # --- The bot checks for math functions ---
    if any(token in math_functions for token in tokens):
        try:
            func_name = [token for token in tokens if token in math_functions][0]
            args = [float(token) for token in tokens if token.replace('.', '', 1).isdigit()]
            result = math_functions[func_name](*args)
            return f"The result is {result}"
        except Exception as e:
            return f"Error while calculating: {e}"

    # --- Check for geometry problems ---
    for func_name, func in geometry_functions.items():
        if func_name in input_str:
            try:
                args = re.findall(r'\d+\.\d+|\d+', input_str)
                args = [float(arg) for arg in args]
                result = func(*args)
                return f"The result is {result}"
            except Exception as e:
                return f"Error while calculating {func_name}: {e}"

    # --- Check for arithmetic expressions ---
    if re.search(r'[\+\-\*/\^]', input_str):
        expr = input_str.replace('^', '**')  # allow ^ for power
        result = safe_eval(expr)
        if result is not None:
            return f"The result is {result}"
        else:
            return "Sorry, I couldn't evaluate that arithmetic expression."

    return "Sorry, I didn't understand that."

def main():
    while True:
        user_input = input("Enter a math problem or type 'quit' to exit: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        print(process_input(user_input))

if __name__ == "__main__":
    main()
