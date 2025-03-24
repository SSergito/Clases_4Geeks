# Your code here
def divisible_binary(sequency):
    bin_numbers = sequency.split(",")
    result = [number for number in bin_numbers if int(number,2)%5==0]
    return ",".join(result)

print(divisible_binary("1111,1000,0101,0000"))