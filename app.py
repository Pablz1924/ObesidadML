import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# 1. Cargar datos
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    return df

data = load_data()

st.title("🧠 Clasificador de Obesidad con Perceptrón")

st.write("Dataset cargado con éxito:")
st.write(data.head())

# ------------------------------------------------------
# 2. Preparar datos (ajusta según tu dataset)
# ------------------------------------------------------
# Suponiendo que la columna objetivo se llama "NObeyesdad"
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# Convertir categóricas en numéricas (si las hay)
X = pd.get_dummies(X)

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------
# 3. Entrenar perceptrón
# ------------------------------------------------------
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------------------------------
# 4. Inputs del usuario
# ------------------------------------------------------
st.header("Introduce tus datos:")

# Aquí pon los inputs reales de tu dataset, ejemplo:
genero = st.selectbox("Género", ["Male", "Female"])
edad = st.number_input("Edad", min_value=1, max_value=100, value=25)
peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0)
altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)

# Crear dataframe con los datos del usuario
entrada = pd.DataFrame([{
    "Gender": genero,
    "Age": edad,
    "Weight": peso,
    "Height": altura
}])

# Convertir categóricas igual que en el dataset
entrada = pd.get_dummies(entrada)
entrada = entrada.reindex(columns=X.columns, fill_value=0)

# ------------------------------------------------------
# 5. Predicción
# ------------------------------------------------------
if st.button("Predecir"):
    pred = clf.predict(entrada)[0]
    st.success(f"Predicción: {pred}")
