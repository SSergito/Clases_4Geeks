from flask import Flask, render_template, request
from pickle import load
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado y el scaler
with open('random_forest_model.sav', 'rb') as file:
    model = load(file)

with open('scaler.sav', 'rb') as file:
    scaler = load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])
        
        # Crear array para el escalado
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                             insulin, bmi, diabetes_pedigree, age]])
        
        # Escalar los datos
        scaled_features = scaler.transform(features)
        
        # Hacer predicción
        prediction = model.predict(scaled_features)
        outcome = int(prediction[0])
        
        # Mostrar resultado
        return render_template('index.html', prediction=outcome)
    
    except Exception as e:
        # Manejo de errores
        error_message = f"Ocurrió un error: {str(e)}"
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)