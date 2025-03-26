def hours_minutes(seconds):
  # Your code here
  horas = seconds//3600
  minutos = (seconds%3600)//60
  segundos = (seconds%3600)%60


  return horas, minutos, segundos

# Invoke the function and pass any integer as its argument
print(hours_minutes(3905))

