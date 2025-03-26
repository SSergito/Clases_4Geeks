# Your code here
def squares_dictionary(n):
    result = {}
    for i in range(n+1):
        if i:
            result[i] = i*i
    return result

print(squares_dictionary(8))