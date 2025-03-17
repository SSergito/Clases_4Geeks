# Your code here
def square_odd_numbers(sequency):
    return [int(num)**2 for num in sequency.split(",") if int(num) % 2]

print(square_odd_numbers("1,2,3,4,5,6,7,8,9"))