from utils import db_connect
engine = db_connect()

# Import
# Basics
import pandas as pd
import numpy as np

# Visualizacion
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ESCALAMIENTO
from sklearn.preprocessing import MinMaxScaler

# MODELOS
from sklearn.ensemble import RandomForestClassifier

# METRICAS
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# GUARDADO DEL MODELO
from pickle import dump

# Get data
train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/sergio-roque-decision-tree/refs/heads/main/data/processed/train_data.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/sergio-roque-decision-tree/refs/heads/main/data/processed/test_data.csv")

# Separacion de datos en train y test 
X_train = train_data.drop("Outcome", axis=1)
y_train = train_data["Outcome"]
X_test = test_data.drop("Outcome", axis=1)
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

# Mantengo los datos sin seleccionar parametros relevantes
X_test_sel = X_test_scal.copy()
X_train_sel = X_train_scal.copy()

# Modelo con parametros optimizados encontrados con combinaciones de grid search y algunas ejecuciones de random search
model = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=5, max_features=None, min_samples_leaf=1, min_samples_split=10, random_state = 42)

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
print(25*"-")
print("Accuracy Train: ", accuracy_train)
print("F1 score Train: ", f1_score_train)
print("Precision Train: ", precision_train)
print("Recall Train: ", recall_train)
print(25*"-")

cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 6))  # Ajusta el tamaño si quieres
disp.plot(ax=ax, cmap='Blues')

# Guarda la imagen como PNG
plt.savefig("graph/matriz_confusion.png", dpi=300, bbox_inches='tight')
plt.close()

y_proba = model.predict_proba(X_test_sel)[:, 1]
# ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot mejorado
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Línea aleatoria
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("graph/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

dump(model, open("models/random_forest_optimized_42.sav", "wb"))