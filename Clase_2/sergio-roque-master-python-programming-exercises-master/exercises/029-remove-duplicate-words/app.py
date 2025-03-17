# Your code here
def remove_duplicate_words(phrase):
    new_set = set(phrase.split())
    lista = list(new_set)
    lista.sort()
    return " ".join(lista)

print(remove_duplicate_words("hello world and practice makes perfect and hello world again"))