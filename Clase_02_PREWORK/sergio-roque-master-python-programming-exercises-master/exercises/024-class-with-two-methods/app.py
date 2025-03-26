# Your code here
class InputOutString():
    def __init__(self):
        self.mi_string = ""

    def get_string(self):
        self.mi_string = input("Escriba el texto: ")
        return self.mi_string

    def print_string(self):
        print(self.mi_string.upper())

mi_clase = InputOutString()
mi_clase.get_string()
mi_clase.print_string()
