import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_obesidad.pkl")

st.set_page_config(page_title="Predicción de Obesidad", layout="centered")

st.title("🧑‍⚕️ Predicción de Obesidad")
st.write("Ingrese sus datos para predecir la categoría de obesidad según el modelo entrenado.")

# ==========================
# Inputs del usuario
# ==========================

# Atributos físicos
gender = st.selectbox("Género", ["Male", "Female"])
age = st.slider("Edad", 5, 100, 25)
height = st.slider("Altura (m)", 1.20, 2.20, 1.70)
weight = st.slider("Peso (kg)", 30.0, 200.0, 70.0)

# Historial familiar
family_history = st.selectbox("Historial familiar con sobrepeso", ["yes", "no"])

# Hábitos alimenticios
favc = st.selectbox("Consumo frecuente de comida alta en calorías (FAVC)", ["yes", "no"])
fcvc = st.slider("Frecuencia de consumo de vegetales (FCVC)", 0.0, 3.0, 1.0)
ncp = st.slider("Número de comidas principales al día (NCP)", 1, 6, 3)
caec = st.selectbox("Consumo entre comidas (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
ch2o = st.slider("Consumo de agua diario (litros) (CH2O)", 0.0, 5.0, 2.0)
calc = st.selectbox("Consumo de alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("¿Fuma? (SMOKE)", ["yes", "no"])
scc = st.selectbox("Monitorea su consumo de calorías (SCC)", ["yes", "no"])

# Actividad física y tecnología
faf = st.slider("Frecuencia de actividad física (FAF)", 0.0, 3.0, 1.0)
tue = st.slider("Tiempo en dispositivos tecnológicos (horas) (TUE)", 0.0, 24.0, 4.0)

# Transporte
mtrans = st.selectbox("Medio de transporte (MTRANS)", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

# ==========================
# Procesar inputs
# ==========================

def Categorico_a_numerico(valor, categorias):
    return categorias.index(valor)

# Mapear valores categóricos
gender_val = Categorico_a_numerico(gender, ["Female", "Male"])
family_val = Categorico_a_numerico(family_history, ["no", "yes"])
favc_val = Categorico_a_numerico(favc, ["no", "yes"])
caec_val = Categorico_a_numerico(caec, ["no", "Sometimes", "Frequently", "Always"])
calc_val = Categorico_a_numerico(calc, ["no", "Sometimes", "Frequently", "Always"])
smoke_val = Categorico_a_numerico(smoke, ["no", "yes"])
scc_val = Categorico_a_numerico(scc, ["no", "yes"])
mtrans_val = Categorico_a_numerico(mtrans, ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Crear vector de entrada
entrada = np.array([[gender_val, age, height, weight, 
                     family_val, favc_val, fcvc, ncp, caec_val, 
                     smoke_val, ch2o, scc_val, faf, tue, 
                     calc_val, mtrans_val]])

# ==========================
# Predicción
# ==========================
if st.button("Predecir"):
    pred = modelo.predict(entrada)
    st.success(f"📊 Resultado de la predicción: **{pred[0]}**")
