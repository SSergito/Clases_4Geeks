# Your code here
def matrix_builder(number):
    matrix = []
    matrix_rows = []
    for i in range(number):
        matrix_rows.append(1)
    for i in range(number):
        matrix.append(matrix_rows)
    return matrix

print(matrix_builder(5))