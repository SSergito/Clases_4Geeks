# Complete the function to return the total cost in dollars and cents of (n) cupcakes
def total_cost(d, c, n):
    return (d*n + c*n//100, c*n%100)


# Invoke the function with three integers: total_cost(dollars, cents, number_of_cupcakes)
print(total_cost(15,22,4))

