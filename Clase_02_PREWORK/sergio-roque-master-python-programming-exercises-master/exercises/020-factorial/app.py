# Your code here
def factorial(num):
    # resultado = 1
    # while num > 1:
    #     resultado *= num
    #     num -= 1
    # return resultado
    resultado = 1
    for i in range(num):
        resultado *= num
        num -= 1
    return resultado


print(factorial(8))