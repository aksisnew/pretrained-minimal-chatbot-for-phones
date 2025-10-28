import nltk
from nltk.tokenize import word_tokenize
import math
import re
import ast
import operator as op
import random

# Initialize NLTK
nltk.download('punkt', quiet=True)

# Safe arithmetic evaluation
operators = {
    # Arithmetic
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,

    # Unary
    ast.UAdd: op.pos,
    ast.USub: op.neg,

    # Bitwise (optional — remove if you don't want them)
    ast.BitXor: op.xor,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.Invert: op.invert,
    ast.LShift: op.lshift,
    ast.RShift: op.rshift,

    # Comparisons (for evaluating logical statements)
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,

    # Boolean
    ast.And: op.and_,
    ast.Or: op.or_
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
def trig_in_degrees(func, x, decimals=8):
    """
    Computes a trigonometric function in degrees.
    
    Parameters:
        func: math.sin, math.cos, math.tan, etc.
        x: float or int (angle in degrees) or a list of angles
        decimals: number of decimals to round the result

    Returns:
        float or list of floats
    """
    # Handle list input
    if isinstance(x, (list, tuple)):
        return [trig_in_degrees(func, angle, decimals) for angle in x]

    # Validate input
    try:
        angle = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid input for trigonometric calculation: {x}")

    radians = math.radians(angle)
    result = func(radians)

    # Round to avoid floating point artifacts
    return round(result, decimals)

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
    # --- Circle ---
    'area of circle': lambda r: math.pi * r ** 2,
    'circumference of circle': lambda r: 2 * math.pi * r,

    # --- Rectangle / Square ---
    'area of rectangle': lambda l, w: l * w,
    'perimeter of rectangle': lambda l, w: 2 * (l + w),
    'area of square': lambda a: a ** 2,
    'perimeter of square': lambda a: 4 * a,

    # --- Triangle ---
    'area of triangle': lambda b, h: 0.5 * b * h,
    'perimeter of triangle': lambda a, b, c: a + b + c,
    'herons formula': lambda a, b, c: math.sqrt((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)),

    # --- Sphere ---
    'volume of sphere': lambda r: (4 / 3) * math.pi * r ** 3,
    'surface area of sphere': lambda r: 4 * math.pi * r ** 2,

    # --- Cylinder ---
    'volume of cylinder': lambda r, h: math.pi * r ** 2 * h,
    'surface area of cylinder': lambda r, h: 2 * math.pi * r * (r + h),

    # --- Cone ---
    'volume of cone': lambda r, h: (1 / 3) * math.pi * r ** 2 * h,
    'surface area of cone': lambda r, l: math.pi * r * (r + l),

    # --- Cube ---
    'volume of cube': lambda a: a ** 3,
    'surface area of cube': lambda a: 6 * a ** 2,

    # --- Rectangular Prism / Box ---
    'volume of rectangular prism': lambda l, w, h: l * w * h,
    'surface area of rectangular prism': lambda l, w, h: 2 * (l*w + w*h + h*l),

    # --- Pythagoras ---
    'pythagoras': lambda a, b: math.sqrt(a ** 2 + b ** 2),

    # --- ADDED ---
    'area of parallelogram': lambda b, h: b * h,
'area of rhombus': lambda d1, d2: 0.5 * d1 * d2,
'area of trapezium': lambda a, b, h: 0.5 * (a + b) * h,
'area of polygon': lambda n, s: (n * s ** 2) / (4 * math.tan(math.pi / n)),


    # --- Circle Sector / Segment ---
    'area of sector': lambda r, angle: (angle / 360) * math.pi * r ** 2,
    'arc length': lambda r, angle: (angle / 360) * 2 * math.pi * r
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
    greetings = [
    "Hey there! 👋 What math problem shall we tackle?",
    "Hi! Ready to do some quick math?",
    "Hello! Let’s solve something fun today.",
    "Welcome! Got a math question for me?",
    "Yo! 😎 Time to crunch some numbers!",
    "Greetings, math explorer! Ready to calculate?",
    "Hiya! Let’s see what math magic we can do today.",
    "Hello there! Numbers await your command!",
    "Hey! Want to challenge me with some math?",
    "Hi! Let’s turn those numbers into answers.",
    "Salutations! Ready for some brain exercise?",
    "Hey friend! Got a tricky math puzzle for me?",
    "Howdy! Let’s tackle some numbers together.",
    "Hi! Math awaits, and I’m here to help.",
    "Hello! Let’s see what kind of math adventure we can go on today!"
]


   thinking = [
    "Hmm... let me calculate that 🤔",
    "Alright, crunching the numbers... 🧮",
    "One sec, working it out...",
    "Let’s see what the math says...",
    "Just a moment, let me figure this out... 🧐",
    "Crunching some numbers in my head...",
    "Hold on, running the calculations...",
    "Thinking… numbers are aligning... 🔢",
    "Let me solve this puzzle for you...",
    "Working my math magic... ✨",
    "Hmm… let’s see… carry the one…",
    "Calculating… almost there…",
    "Processing your math problem… ⚙️",
    "Let me double-check my numbers…",
    "Hmm, this looks interesting… solving now!"
]


   farewells = [
    "Goodbye! Keep being awesome at math! 👋",
    "See you later, math whiz! ✨",
    "Bye! Don’t forget to challenge your brain again soon.",
    "Take care! Numbers never sleep. 😄",
    "Farewell! May your equations always balance.",
    "Catch you later! Keep crunching those numbers.",
    "Bye-bye! Hope your math adventures continue!",
    "See ya! Remember, math is everywhere!",
    "Adios! Keep solving and stay curious.",
    "Goodbye! May your calculations always be correct.",
    "Until next time! Don’t forget to have fun with math!",
    "Bye! Keep those neurons firing! 🔥",
    "Take it easy! Math will be waiting when you return.",
    "See you soon! Keep exploring the world of numbers.",
    "Goodbye! Stay sharp, math master! 🧠"
]


    print(random.choice(greetings))

    while True:
        user_input = input("\nEnter a math question (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print(random.choice(farewells))
            break

        print(random.choice(thinking))
        print(process_input(user_input))

if __name__ == "__main__":
    main()
