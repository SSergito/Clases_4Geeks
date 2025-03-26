# Your code here
def net_amount(register):
    amount = 0
    for index,value in enumerate(register.split()):
        if value == "D":
            amount += int(register.split()[index+1])
        elif value == "W":
            amount -= int(register.split()[index+1])
    return amount

print(net_amount("D 300 D 300 W 200 D 100"))