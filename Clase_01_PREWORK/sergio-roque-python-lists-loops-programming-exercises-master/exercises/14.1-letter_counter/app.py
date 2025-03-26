par = "Lorem ipsum dolor sit amet consectetur adipiscing elit Curabitur eget bibendum turpis Curabitur scelerisque eros ultricies venenatis mi at tempor nisl Integer tincidunt accumsan cursus"

counts = {}

# Your code here
for i in par.lower():
    if i != " " and i in counts:
        counts[i] += 1
    elif i != " ":
        counts[i] = 1

print(counts)