# Your code here
def all_digits_even():
    even = [str(num) for num in range(1000,3001) if not num%2]
    return ",".join(even)

print(all_digits_even())