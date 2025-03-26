# Your code here
class MySevenNumbers:
    def __init__(self):
        pass
    def divisible_numbers(self, n):
        for i in range(n+1):
            if i % 7 == 0:
                yield i

prueba = MySevenNumbers()
print(list(prueba.divisible_numbers(548)))