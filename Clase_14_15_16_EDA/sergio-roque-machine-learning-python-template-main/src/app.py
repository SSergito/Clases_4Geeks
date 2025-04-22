import pandas as pd
from utils import db_connect
engine = db_connect()

# Me traigo los datos de la url proporcionada y los guardo localmente en mi proyecto
# url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"
# df = pd.read_csv(url)
# df.to_csv("data/raw/data.csv", index=False)

df = pd.read_csv("data/raw/data.csv")
print(df.head())