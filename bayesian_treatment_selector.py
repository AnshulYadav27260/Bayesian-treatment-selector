import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import streamlit as st

# Load precomputed dataset (Simulated Data for Training Deep Learning Model with Additional Predictors)
def generate_patient_data(num_samples=500):
    np.random.seed(42)
    return pd.DataFrame({
        "IL-6 Reduction (%)": np.random.normal(-13.0, 1.5, num_samples),
        "TNF-α Reduction (%)": np.random.normal(-14.5, 1.2, num_samples),
        "Dopaminergic Neuron Survival (%)": np.random.normal(3.0, 0.8, num_samples),
        "Genetic Risk Score": np.random.normal(0.5, 0.2, num_samples),  # New Feature
        "MRI Brain Atrophy (%)": np.random.normal(2.5, 1.0, num_samples),  # New Feature
        "Comorbidity Score": np.random.normal(3.0, 1.5, num_samples),  # New Feature
        "MDS-UPDRS III Improvement": np.random.normal(-4.5, 0.9, num_samples),
        "Treatment": np.random.choice(["Semaglutide", "Liraglutide", "Exenatide"], num_samples)
    })

data = generate_patient_data()

# Encode Treatment as a Numeric Variable
data["Treatment_Label"] = data["Treatment"].astype('category').cat.codes

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = data[["IL-6 Reduction (%)", "TNF-α Reduction (%)", "Dopaminergic Neuron Survival (%)", "Genetic Risk Score", "MRI Brain Atrophy (%)", "Comorbidity Score"]]
y = data["Treatment_Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Enhanced Deep Learning Model with Additional Features
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(6,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 Treatments
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Streamlit UI for Deep Learning-Enhanced AI Tool with Additional Predictors
st.title("AI-Assisted Deep Learning Treatment Selector with Genetic & Imaging Data")

st.sidebar.header("Enter Patient Biomarker & Clinical Data")
il6_reduction = st.sidebar.slider("IL-6 Reduction (%)", -20.0, 0.0, -13.0)
tnf_reduction = st.sidebar.slider("TNF-α Reduction (%)", -20.0, 0.0, -14.5)
dopamine_survival = st.sidebar.slider("Dopaminergic Neuron Survival (%)", 0.0, 10.0, 3.0)
genetic_risk = st.sidebar.slider("Genetic Risk Score", 0.0, 1.0, 0.5)
mri_atrophy = st.sidebar.slider("MRI Brain Atrophy (%)", 0.0, 10.0, 2.5)
comorbidity_score = st.sidebar.slider("Comorbidity Score", 0.0, 10.0, 3.0)

# Prepare patient data for model prediction
patient_input = np.array([[il6_reduction, tnf_reduction, dopamine_survival, genetic_risk, mri_atrophy, comorbidity_score]])
predicted_treatment = model.predict(patient_input)
selected_treatment = ["Semaglutide", "Liraglutide", "Exenatide"][np.argmax(predicted_treatment)]

st.subheader("Deep Learning-Based Recommended Treatment with Genetic & Imaging Data")
st.write(f"**Optimal Treatment: {selected_treatment}**")
