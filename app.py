import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Título
# ------------------------------------------------------
st.title("🧠 Clasificador de Obesidad con Perceptrón")

# ------------------------------------------------------
# 2. Inputs del usuario
# ------------------------------------------------------
st.header("Introduce tus datos:")

edad = st.number_input("Edad", min_value=1, max_value=100, value=25)
peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0)
altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)

# Calculamos IMC
imc = peso / (altura ** 2)
st.write(f"Tu IMC es: **{imc:.2f}**")

# ------------------------------------------------------
# 3. Entrenamiento del modelo (ejemplo simplificado)
# ------------------------------------------------------
# Aquí pondrías tu dataset real. Para demo uso datos random:
X, y = np.random.rand(200, 3), np.random.randint(0, 2, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------------------------------
# 4. Predicción con datos del usuario
# ------------------------------------------------------
if st.button("Predecir"):
    entrada = np.array([[edad/100, peso/200, altura/2.5]])  # normalización simple
    pred = clf.predict(entrada)[0]

    if pred == 1:
        st.error("⚠️ Riesgo de obesidad detectado")
    else:
        st.success("✅ Peso dentro de rango normal")
