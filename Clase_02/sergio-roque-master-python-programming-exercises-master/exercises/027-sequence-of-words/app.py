# Your code here
def sequence_of_words(words):
    list_words = words.split(",")
    list_words.sort()
    return ",".join(list_words)

print(sequence_of_words("without,hello,bag,world"))