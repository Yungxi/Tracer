"""Example file to test the tracer."""


def add_numbers(a, b):
    """Add two numbers together."""
    return a + b


def multiply(x, y):
    """Multiply two numbers."""
    return x * y


def divide(a, b):
    """Divide a by b."""
    return a / b


def greet(name):
    """Generate a greeting message."""
    return f"Hello, {name}!"


# Main process
result1 = add_numbers(5, 3)
print(f"5 + 3 = {result1}")

result2 = multiply(4, 7)
print(f"4 * 7 = {result2}")

result3 = divide(20, 4)
print(f"20 / 4 = {result3}")

message = greet("World")
print(message)
