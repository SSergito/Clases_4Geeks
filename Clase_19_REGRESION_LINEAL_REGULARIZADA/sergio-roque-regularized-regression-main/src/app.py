from utils import db_connect
engine = db_connect()

# your code here
# BASICS
import pandas as pd
import numpy as np

# ESCALAMIENTO
from sklearn.preprocessing import MinMaxScaler

# SELECCION DE PARAMETROS
from sklearn.feature_selection import mutual_info_regression, SelectKBest

# MODELOS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV

# METRICAS
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

X_train = train_data.drop(["Heart disease_prevalence"], axis=1)
y_train = train_data["Heart disease_prevalence"]

X_test = test_data.drop(["Heart disease_prevalence"], axis=1)
y_test = test_data["Heart disease_prevalence"]

# Escalado
# Instancio el escalador
scaler = MinMaxScaler()

# Entreno el escalador con los datos de entrenamiento
scaler.fit(X_train)

# Aplico el escalador en ambos
X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = X_train.columns)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = X_test.columns)

# Seleccion de parametros
selection_model = SelectKBest(mutual_info_regression, k = 5)
selection_model.fit(X_train_scal, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_scal), columns = X_train_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_scal), columns = X_test_scal.columns.values[ix])

# >>>>>>>>>>>>>>> APLICACION DE MODELOS <<<<<<<<<<<<<<<
# Con regresion lineal tipica
# Modelo
reg_lin = LinearRegression()

# Entrenamiento
reg_lin.fit(X_train_sel, y_train)

# Predicción
y_pred = reg_lin.predict(X_test_sel)
y_pred

# Metricas
mse_rl = mean_squared_error(y_test, y_pred)
rmse_rl = np.sqrt(mse_rl)
r2_rl = r2_score(y_test, y_pred)

print("Con Modelo de Regresion Lineal tipica:")
print("MSE: ", mse_rl)
print("RMSE: ", rmse_rl)
print("Coeficiente de determinación: ", r2_rl)

# Analisis de overfitting
# Predicciones
y_train_pred = reg_lin.predict(X_train_sel)
y_test_pred = reg_lin.predict(X_test_sel)

# Métricas en train
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))

# Métricas en test
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

# Conclusion: no esta haciendo overfitting

# Regularizacion Lasso
# modelo
reg_lasso = Lasso(alpha = 0.3, max_iter = 1000)

# entrenamiento
reg_lasso.fit(X_train_sel, y_train)

# prediccion
y_pred = reg_lasso.predict(X_test_sel)
y_pred

# metricas
mse_l1 = mean_squared_error(y_test, y_pred)
rmse_l1 = np.sqrt(mse_l1)
r2_l1 = r2_score(y_test, y_pred)

print("\nCon Modelo de Regresion Lineal Regulación Lasso:")
print("MSE: ", mse_l1)
print("RMSE: ", rmse_l1)
print("Coeficiente de determinación: ", r2_l1)

# Por el coeficiente de determinacion tan bajo q obtuve decido hacer cross validation con Lasso para buscar mejor alpha

model = LassoCV(cv=5).fit(X_train_sel, y_train)
print("Mejor alpha:", model.alpha_)

# Regularizacion Lasso con CV
# modelo
reg_lasso = Lasso(alpha = model.alpha_, max_iter = 1000)

# entrenamiento
reg_lasso.fit(X_train_sel, y_train)

# prediccion
y_pred = reg_lasso.predict(X_test_sel)
y_pred

# metricas
mse_l1 = mean_squared_error(y_test, y_pred)
rmse_l1 = np.sqrt(mse_l1)
r2_l1 = r2_score(y_test, y_pred)

print("\nCon Modelo de Regresion Lineal Regulación LassoCV:")
print("MSE: ", mse_l1)
print("RMSE: ", rmse_l1)
print("Coeficiente de determinación: ", r2_l1)

# Analisis de overfitting
# Predicciones
y_train_pred = reg_lasso.predict(X_train_sel)
y_test_pred = reg_lasso.predict(X_test_sel)

# Métricas en train
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))

# Métricas en test
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

# Conclusion: no esta haciendo overfitting

# Regularizacion Ridge
# modelo
reg_ridge = Ridge(alpha = 0.2, max_iter = 4000)

# entrenamiento
reg_ridge.fit(X_train_sel, y_train)

# prediccion
y_pred = reg_ridge.predict(X_test_sel)
y_pred

# metricas
mse_l2 = mean_squared_error(y_test, y_pred)
rmse_l2 = np.sqrt(mse_l2)
r2_l2 = r2_score(y_test, y_pred)

print("\nCon Modelo de Regresion Lineal Regulación Ridge:")
print("MSE: ", mse_l2)
print("RMSE: ", rmse_l2)
print("Coeficiente de determinación: ", r2_l2)

# Analisis de overfitting
# Predicciones
y_train_pred = reg_ridge.predict(X_train_sel)
y_test_pred = reg_ridge.predict(X_test_sel)

# Métricas en train
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))

# Métricas en test
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

# Conclusion: no esta haciendo overfitting

# Comparacion de la aplicacion de los 3 modelos de regresion
# Datos
data = {
    'Modelo': ['Regresión lineal', "Regulación Lasso CV", "Regulación Ridge"],
    'MSE': [mse_rl, mse_l1, mse_l2],
    'RMSE': [rmse_rl, rmse_l1, rmse_l2],
    'R²': [r2_rl, r2_l1, r2_l2]
}

# Crear el DataFrame
resultados = pd.DataFrame(data)

# Mostrar el DataFrame
print()
print(resultados)