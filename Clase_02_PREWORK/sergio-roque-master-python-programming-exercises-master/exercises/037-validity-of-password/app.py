# Your code here
import re

password = "ABd1234@1"

def valid_password(password):
    reg_expressions = [r"[A-Z]", r"[a-z]", r"\d", r"[$#@]", r"^.{6,12}$"]
    valid = []
    for i in reg_expressions:
        if re.search(i, password):
            valid.append(True)
        else: valid.append(False)
    if False in valid:
        return "Invalid password. Please try again"
    else: return "Valid password"

print(valid_password(password))
