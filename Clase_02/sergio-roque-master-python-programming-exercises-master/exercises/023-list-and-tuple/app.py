# Your code here
def list_and_tuple(*args):
    lista = []
    for i in args:
        lista.append(str(i))

    return lista, tuple(lista)

mi_lista, mi_tupla = list_and_tuple(34,67,55,33,12,98)
print(mi_lista)
print(mi_tupla)