import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
model = joblib.load('model.joblib')
st.title("Prédiction des primes d'assurance maladie")
# Récupérer les informations
age = st.number_input("Entrez votre âge", min_value=18, max_value=100, step=1)
bmi = st.number_input("Entrez votre IMC (Indice de Masse Corporelle)", min_value=10.0, max_value=70.0, step=0.1)
smoker = st.radio("Êtes-vous fumeur ?", options=["Oui", "Non"])
region = st.selectbox("Sélectionnez votre région", options=["northeast", "northwest", "southeast", "southwest"])
data = np.array([[ bmi, age, 1 if smoker=="Oui" else 0, 
                  1 if region=="northeast" else 0, 1 if region=="northwest" else 0, 
                  1 if region=="southeast" else 0, 1 if region=="southwest" else 0]])

# Standardiser les variables numériques
scaler = StandardScaler()
new_data = scaler.fit_transform(data[:, [0, 1]])
data[:, [0, 1]] = new_data

if st.button("Calculer la prime"):
    prediction = model.predict(data)[0]
    st.success(f"Votre prime d'assurance maladie estimée est de ${prediction:.2f}")
