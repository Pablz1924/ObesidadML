import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# 1. Cargar dataset
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

data = load_data()

st.title("🧠 Clasificador de Obesidad con Perceptrón")

# ------------------------------------------------------
# 2. Preparar datos
# ------------------------------------------------------
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# Codificar variables categóricas
X = pd.get_dummies(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------------------------------
# 3. Inputs del usuario
# ------------------------------------------------------
st.header("Introduce tus datos")

# Variables físicas
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

# Hábitos alimenticios
favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
fcvc = st.slider("Frequency of consumption of vegetables (FCVC)", 1, 3, 2)
ncp = st.slider("Number of main meals (NCP)", 1, 4, 3)
caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
ch2o = st.slider("Consumption of water daily (CH2O)", 1, 3, 2)
calc = st.selectbox("Consumption of alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("SMOKE", ["yes", "no"])
scc = st.selectbox("Calories consumption monitoring (SCC)", ["yes", "no"])

# Actividad física y tecnología
faf = st.slider("Physical activity frequency (FAF)", 0, 3, 1)
tue = st.slider("Time using technology devices (TUE)", 0, 2, 1)

# Transporte
mtrans = st.selectbox("Transportation used (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

# Historial familiar
family_history = st.selectbox("Family history with overweight", ["yes", "no"])

# --------------------------------

# ------------------------------------------------------
# 4. Armar dataframe de entrada
# ------------------------------------------------------
entrada = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "CH2O": ch2o,
    "CALC": calc,
    "SMOKE": smoke,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "MTRANS": mtrans,
    "family_history_with_overweight": family_history
}])

# One-hot encoding igual que en el dataset
entrada = pd.get_dummies(entrada)
entrada = entrada.reindex(columns=X.columns, fill_value=0)

# ------------------------------------------------------
# 5. Predicción
# ------------------------------------------------------
if st.button("Predecir"):
    pred = clf.predict(entrada)[0]
    st.success(f"Categoría de obesidad predicha: **{pred}**")
