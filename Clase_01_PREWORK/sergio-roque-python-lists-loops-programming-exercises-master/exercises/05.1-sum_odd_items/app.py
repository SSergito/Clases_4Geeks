my_list = [4,5,734,43,45,100,4,56,23,67,23,58,45]

# Your code here
def sum_odds(lista):
    suma = 0
    for i in lista:
        if i%2:
            suma+=i
    return suma

print(sum_odds(my_list))