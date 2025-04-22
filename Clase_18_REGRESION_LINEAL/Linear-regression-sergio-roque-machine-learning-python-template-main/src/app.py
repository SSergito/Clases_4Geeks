from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

X_train = train_data.drop(["charges"], axis=1)
y_train = train_data["charges"]

X_test = test_data.drop(["charges"], axis=1)
y_test = test_data["charges"]

# Escalado
# Instancio el escalador
scaler = StandardScaler()

# Entreno el escalador con los datos de entrenamiento
scaler.fit(X_train)

# Aplico el escalador en ambos
X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = X_train.columns)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = X_test.columns)

# Genera el modelo
model = LinearRegression()

# Entrena
model.fit(X_train_scal, y_train)

# Prediccion
y_pred = model.predict(X_test_scal)

print(f"Intercepto (b0): {model.intercept_}")
print(f"Coeficientes (b1): {model.coef_}")

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred):.4f}")

print(f"Raíz del error cuadrático medio: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")