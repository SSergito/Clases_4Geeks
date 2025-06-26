import streamlit as st
import pickle
import numpy as np

# Cargar modelos y transformadores
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Cargar encoders
label_encoder_gender = load_pickle('label_encoder_gender.sav')
label_encoder_suicidal = load_pickle('label_encoder_suicidal.sav')
label_encoder_mental = load_pickle('label_encoder_mental.sav')
ordinal_encoder_sleep = load_pickle('ordinal_encoder_sleep.sav')
ordinal_encoder_dietary = load_pickle('ordinal_encoder_dietary.sav')

# Cargar scaler y selección de características
scaler = load_pickle('scaler.sav')
selector = load_pickle('selection_model.sav')

# Cargar modelo
model = load_pickle('logistic_regression_optimized_42.sav')

# Interfaz de usuario
st.title("Predicción de Salud Mental Estudiantil")

age = st.slider("Edad", 18, 34)
academic_pressure = st.slider("Presión académica (0-5)", 0, 5)
cgpa = st.slider("CGPA (Promedio Acumulado de Puntos) (5-10)", 5.0, 10.0, step=0.1)
study_satisfaction = st.slider("Satisfacción con el estudio (0-5)", 0, 5)
study_hours = st.slider("Horas de estudio/trabajo (0-12)", 0, 12)
financial_stress = st.slider("Estrés financiero (1-5)", 1, 5)

gender = st.selectbox("Género", ["Male", "Female"])
suicidal = st.selectbox("¿Has tenido pensamientos suicidas?", ["Yes", "No"])
mental_illness = st.selectbox("Antecedentes familiares de enfermedad mental", ["Yes", "No"])
sleep_options = {
    "Menos de 5 horas": "'Less than 5 hours'",
    "5-6 horas": "'5-6 hours'",
    "7-8 horas": "'7-8 hours'",
    "Más de 8 horas": "'More than 8 hours'"
}
sleep_label = st.selectbox("Duración del sueño", list(sleep_options.keys()))
sleep_duration = sleep_options[sleep_label]
dietary = st.selectbox("Hábitos alimenticios", ["Unhealthy", "Moderate", "Healthy"])

if st.button("Predecir"):
    # Codificar entradas
    gender_encoded = label_encoder_gender.transform([gender])[0]
    suicidal_encoded = label_encoder_suicidal.transform([suicidal])[0]
    mental_encoded = label_encoder_mental.transform([mental_illness])[0]
    sleep_encoded = ordinal_encoder_sleep.transform([[sleep_duration]])[0][0]
    dietary_encoded = ordinal_encoder_dietary.transform([[dietary]])[0][0]

    # Construir array de entrada
    input_data = np.array([[age, academic_pressure, cgpa, study_satisfaction,
                            study_hours, financial_stress, gender_encoded,
                            suicidal_encoded, mental_encoded, sleep_encoded,
                            dietary_encoded]])

    # Escalar
    input_scaled = scaler.transform(input_data)

    # Selección de características
    input_selected = selector.transform(input_scaled)

    # Predicción
    prediction = model.predict(input_selected)[0]

    if prediction == 0:
        st.success("✅ No se predicen indicios de depresión. ¡Sigue cuidando tu salud mental! 😊")
        st.balloons()
    else:
        st.error("⚠️ Se predicen posibles síntomas de depresión. Considera hablar con un profesional o pedir apoyo. 💬")
        st.markdown("**Recuerda:** No estás solo/a. Buscar ayuda es un acto de valentía. ❤️")

