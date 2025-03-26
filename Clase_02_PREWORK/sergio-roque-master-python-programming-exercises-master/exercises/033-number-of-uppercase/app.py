# Your code here
def number_of_uppercase(sentence):
    counter = {"UPPERCASE": 0, "LOWERCASE": 0}
    for letter in sentence:
        if letter.isupper():
            counter["UPPERCASE"] += 1
        elif letter.islower():
            counter["LOWERCASE"] += 1
    return f"UPPERCASE {counter['UPPERCASE']}\nLOWERCASE {counter['LOWERCASE']}"

print(number_of_uppercase("Hello world!"))