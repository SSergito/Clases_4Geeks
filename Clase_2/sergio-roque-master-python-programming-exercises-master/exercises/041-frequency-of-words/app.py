# Your code here
def compute_word_frequency(words_list):
    separate_words = words_list.split()
    dict_words = {}
    for i in separate_words:
        if i in dict_words:
            dict_words[i] += 1
        else: dict_words[i] = 1
    return dict_words

my_words = compute_word_frequency("New to Python or choosing between Python 2 and Python 3? Read Python 2 or Python 3.")
for clave, valor in my_words.items():
    print(f"{clave}: {valor}")