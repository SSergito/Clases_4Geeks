# import os
# import seaborn as sns
# import time
from bs4 import BeautifulSoup
import requests
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

def extract_data(url):
    # Peticion a la web para extraer contenido
    response = requests.get(url)
    if response.status_code == 200:
        # Diccionario para almacenar resultados
        data = {"Año":[], "Ingresos":[], "Crecimiento":[]}
        # Parseo del HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Busqueda de la etiqueta tabla y separacion luego de las filas
        tabla = soup.find(class_="table")
        for fila in tabla.find_all("tr"):
            fila_separada = fila.find_all("td")
            if fila_separada:
                # Agrego los datos a mi diccionario
                data["Año"].append(fila_separada[0].get_text(strip=True))
                data["Ingresos"].append(fila_separada[1].get_text(strip=True))
                data["Crecimiento"].append(fila_separada[2].get_text(strip=True))
        # Dataframe con la informacion extraida y luego de limpieza de etiquetas HTML
        df = pd.DataFrame(data)
        print(f"Extraccion finalizada con codigo de respuesta web {response.status_code}")
        return df
    else:
        print(f"Error en la solicitud, codigo {response.status_code}")
        return

def data_2_csv(dataframe):
    # Guardo mi dataframe en csv para las pruebas del codigo
    # Me evito lanzar demasiadas peticiones mientras creo el codigo
    dataframe.to_csv("assets/revenue.csv", index=False)
    print("Dataframe guardado correctamente!!")
    return

url = "https://companies-market-cap-copy.vercel.app/index.html"

df = extract_data(url)
data_2_csv(df)

df = pd.read_csv("assets/revenue.csv")
df["Ingresos"] = df["Ingresos"].str.replace("$", "").str.replace("B", "").str.strip().astype(float)
# No elimino el registro donde Crecimiento contiene un NaN porque al ser el dato mas antiguo el crecimiento es de 0 %
df["Crecimiento"] = df["Crecimiento"].fillna("0%")
df["Crecimiento"] = df["Crecimiento"].str.replace("%", "").str.strip().astype(float)


df = df.rename(columns={"Ingresos":"Ingresos (B)", "Crecimiento": "Crecimiento (%)"})


# PASAR MI DATAFRAME A BASE DE DATOS CON SQL
# BASE DE DATOS GUARDADA EN assets/database.db

connection = sqlite3.connect("assets/database.db")
df.to_sql("ingresos", connection, if_exists="replace", index=False)

# Leyendo mis datos de la base de datos sql
df_read = pd.read_sql("SELECT * FROM ingresos", connection)

connection.close()


# VISUALIZACION DE MIS DATOS CON MATPLOTLIB
# GRAFICOS GUARDADOS EN CARPETA assets/graficos/

# Grafico barras ingresos por año
plt.figure(figsize=(12, 6))
bars = plt.bar(df_read['Año'], df_read['Ingresos (B)'], color='#1f77b4', alpha=0.8)

# Añadir etiquetas con los valores
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}B',
             ha='center', va='bottom', fontsize=9)

plt.title('Evolución de Ingresos Anuales (en Billones USD)')
plt.xlabel('Año')
plt.ylabel('Ingresos (B)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('assets/graficos/ingresos_anuales.png', dpi=300)
plt.close()

# --------------------------------------------------------------------------------
# Grafico lineas evolucion crecimiento anual
plt.figure(figsize=(12, 6))
line = plt.plot(df_read['Año'], df_read['Crecimiento (%)'], marker='o', color='#ff7f0e')

# Añadir etiquetas
for x, y in zip(df_read['Año'], df_read['Crecimiento (%)']):
    plt.text(x, y+5, f'{y:.2f}%', 
             ha='center', 
             color='#804004',
             fontsize=9)

plt.title('Crecimiento Anual (%)')
plt.xlabel('Año')
plt.ylabel('Crecimiento (%)')
plt.grid()
plt.tight_layout()
plt.savefig('assets/graficos/crecimiento_anual.png', dpi=300)
plt.close()


# --------------------------------------------------------------------------------
# Grafico combinado (Barras + Líneas)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Barras (Ingresos)
ax1.bar(df_read['Año'], df_read['Ingresos (B)'], color='#1f77b4', alpha=0.6)
ax1.set_xlabel('Año', fontsize=12)
ax1.set_ylabel('Ingresos (Billones USD)', color='#1f77b4', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(df_read['Año'])
ax1.set_xticklabels(df_read['Año'], rotation=45)

# Línea (Crecimiento)
ax2 = ax1.twinx()
ax2.plot(df_read['Año'], df_read['Crecimiento (%)'], color='#ff7f0e', marker='o', linewidth=2)
ax2.set_ylabel('Crecimiento (%)', color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

# Línea horizontal en y=0 para referencia
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)

plt.title('Relación entre Ingresos y Crecimiento Anual', fontsize=14, pad=20)

# Guardar
fig.tight_layout()
fig.savefig('assets/graficos/combinado_ingresos_crecimiento.png', dpi=300)
plt.close()


# EJERCICIO EXTRA: EXTRAER LAS GANANCIAS DE TESLA EN 2024 DESDE UN NUEVO LINK
url_gain = "https://companies-market-cap-copy.vercel.app/earnings.html"

response_gain = requests.get(url_gain)
if response_gain.status_code == 200:
    soup = BeautifulSoup(response_gain.text, "html.parser")
    tables = soup.find_all("table", class_="table")
    if tables:
        mi_tabla = tables[0]
        filas = mi_tabla.find_all("tr")
        for fila in filas:
            celdas = fila.find_all("td")
            if celdas and celdas[0].get_text(strip=True) == "2024":
                print(f"Ganancias en 2024 de Tesla: {celdas[1].get_text(strip=True)} de dolares")
else:
    print(f"Error en la solicitud, codigo {response_gain.status_code}")