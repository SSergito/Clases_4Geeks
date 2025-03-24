names = ['Alice', 'Bob', 'Marry', 'Joe', 'Hilary', 'Stevia', 'Dylan']

def prepender(name):
    return "My name is: " + name
    
# Your code here
lista_saludos = list(map(prepender,names))
print(lista_saludos)