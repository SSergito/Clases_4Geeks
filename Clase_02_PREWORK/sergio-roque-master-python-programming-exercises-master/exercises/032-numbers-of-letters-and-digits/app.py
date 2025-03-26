# Your code here
def letters_and_digits(sentence):
    counter = {"LETTERS":0,"DIGITS":0}
    for char in sentence:
        if char.isalpha():
            counter["LETTERS"] += 1
        elif char.isdigit():
            counter["DIGITS"] += 1
    return f"LETTERS {counter['LETTERS']}\nDIGITS {counter['DIGITS']}"

print(letters_and_digits("hello world! 123"))