from operator import itemgetter

# Your code here
def sort_tuples_ascending(lista):
    tupled_list = [tuple(i.split(",")) for i in lista]
    datos_ordenados = sorted(tupled_list, key=itemgetter(0,1,2))
    return datos_ordenados


print(sort_tuples_ascending(['Tom,19,80', 'John,20,90', 'Jony,17,91', 'Jony,17,93', 'Jason,21,85']))