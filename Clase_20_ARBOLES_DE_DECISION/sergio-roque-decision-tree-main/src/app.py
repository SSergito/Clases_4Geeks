from utils import db_connect
engine = db_connect()

# your code here
# BASICAS
import pandas as pd
import numpy as np

# ESCALAMIENTO
from sklearn.preprocessing import MinMaxScaler

# SELECCION DE PARAMETROS
from sklearn.feature_selection import chi2, SelectKBest

# MODELOS
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# METRICAS
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score

# GUARDADO DEL MODELO
from pickle import dump

train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

X_train = train_data.drop(["Outcome"], axis=1)
y_train = train_data["Outcome"]

X_test = test_data.drop(["Outcome"], axis=1)
y_test = test_data["Outcome"]

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

# Selección de parámetros
selection_model = SelectKBest(chi2, k = 5)
selection_model.fit(X_train_scal, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_scal), columns = X_train_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_scal), columns = X_test_scal.columns.values[ix])

# >>>>>>>>>> ARBOL DE DECISION CON PARAMETROS POR DEFECTO <<<<<<<<<<
# Modelo
model = DecisionTreeClassifier(random_state = 42)

# Entrenamiento
model.fit(X_train_sel, y_train)

# Predicción
y_pred_test = model.predict(X_test_sel)
y_pred_train = model.predict(X_train_sel)

# Metricas
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)

f1_score_test = f1_score(y_test, y_pred_test, average='micro')
f1_score_train = f1_score(y_train, y_pred_train, average='micro')

precision_test = precision_score(y_test, y_pred_test, average='micro')
precision_train = precision_score(y_train, y_pred_train, average='micro')

recall_test = recall_score(y_test, y_pred_test, average='micro')
recall_train = recall_score(y_train, y_pred_train, average='micro')
print("ARBOL DE DECISION CON PARAMETROS POR DEFECTO")
print("Accuracy Test: ", accuracy_test)
print("F1 score Test: ", f1_score_test)
print("Precision Test: ", precision_test)
print("Recall Test: ", recall_test)
print("-----")
print("Accuracy Train: ", accuracy_train)
print("F1 score Train: ", f1_score_train)
print("Precision Train: ", precision_train)
print("Recall Train: ", recall_train)

dump(model, open("models/decision_tree_classifier_default_42.sav", "wb"))

# >>>>>>>>>> ARBOL DE DECISION CON GRID SEARCH <<<<<<<<<<
print("\nARBOL DE DECISION CON GRID SEARCH")
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # validación cruzada de 5 folds
    scoring='accuracy'  # Métrica a optimizar
)
grid_search.fit(X_train_sel, y_train)
print("Mejores parámetros:", grid_search.best_params_)
# Modelo
model = DecisionTreeClassifier(criterion="entropy", max_depth=5, max_features=None, min_samples_leaf=1, min_samples_split=10, random_state = 42)

# Entrenamiento
model.fit(X_train_sel, y_train)
# Predicción
y_pred_test = model.predict(X_test_sel)
y_pred_train = model.predict(X_train_sel)
# Metricas
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)

f1_score_test = f1_score(y_test, y_pred_test, average='micro')
f1_score_train = f1_score(y_train, y_pred_train, average='micro')

precision_test = precision_score(y_test, y_pred_test, average='micro')
precision_train = precision_score(y_train, y_pred_train, average='micro')

recall_test = recall_score(y_test, y_pred_test, average='micro')
recall_train = recall_score(y_train, y_pred_train, average='micro')

print("Accuracy Test: ", accuracy_test)
print("F1 score Test: ", f1_score_test)
print("Precision Test: ", precision_test)
print("Recall Test: ", recall_test)
print("-----")
print("Accuracy Train: ", accuracy_train)
print("F1 score Train: ", f1_score_train)
print("Precision Train: ", precision_train)
print("Recall Train: ", recall_train)

dump(model, open("models/decision_tree_classifier_optimized_42.sav", "wb"))

# >>>>>>>>>> MODELO DE REGRESION LOGISTICA <<<<<<<<<<
print("\nMODELO DE REGRESION LOGISTICA")
# Modelo
model = LogisticRegression()

# Entrenamiento
model.fit(X_train_sel, y_train)
# Prediccion
y_pred = model.predict(X_test_sel)

print("Accuracy Test: ", accuracy_score(y_test, y_pred))
y_pred_train = model.predict(X_train_sel)
print("Accuracy Train: ", accuracy_score(y_train, y_pred_train))

dump(model, open("models/logistic_regression_default_42.sav", "wb"))

# >>>>>>>>>> MODELO DE REGRESION LOGISTICA CON OPTIMIZACION DE PARAMETROS <<<<<<<<<<
print("\nMODELO DE REGRESION LOGISTICA CON OPTIMIZACION DE PARAMETROS")
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Modelo
model = LogisticRegression()

# Definimos los parámetros que queremos ajustar a mano
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

# Inicializamos la cuadrícula
grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 5)
grid.fit(X_train_sel, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")

model_grid = LogisticRegression(C = 1, penalty="l1", solver = "liblinear")
model_grid.fit(X_train_sel, y_train)

y_pred = model_grid.predict(X_test_sel)
grid_accuracy = accuracy_score(y_test, y_pred)
y_pred_train = model_grid.predict(X_train_sel)

accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)

f1_score_test = f1_score(y_test, y_pred, average='micro')
f1_score_train = f1_score(y_train, y_pred_train, average='micro')

precision_test = precision_score(y_test, y_pred, average='micro')
precision_train = precision_score(y_train, y_pred_train, average='micro')

recall_test = recall_score(y_test, y_pred, average='micro')
recall_train = recall_score(y_train, y_pred_train, average='micro')

print("Accuracy Test: ", accuracy_test)
print("F1 score Test: ", f1_score_test)
print("Precision Test: ", precision_test)
print("Recall Test: ", recall_test)
print("-----")
print("Accuracy Train: ", accuracy_train)
print("F1 score Train: ", f1_score_train)
print("Precision Train: ", precision_train)
print("Recall Train: ", recall_train)

dump(model, open("models/logistic_regression_optimized_42.sav", "wb"))