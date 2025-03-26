# Your code here
def two_dimensional_list(x,y):
    return [[i*j for i in range(y)] for j in range(x)]

print(two_dimensional_list(3,5))