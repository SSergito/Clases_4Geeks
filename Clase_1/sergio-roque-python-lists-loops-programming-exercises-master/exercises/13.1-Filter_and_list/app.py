all_names = ["Romario", "Boby", "Roosevelt", "Emiliy", "Michael", "Greta", "Patricia", "Danzalee"]

# Your code here
resulting_names = list(filter(lambda x: x if x.startswith('R') else None,all_names))


print(resulting_names)




