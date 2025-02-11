!pip install scipy
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta

# Load precomputed Bayesian treatment probabilities
def load_treatment_probabilities():
    return {
        "Semaglutide": 0.765,
        "Liraglutide": 0.694,
        "Exenatide": 0.647
    }

# Function to update Bayesian probabilities dynamically
def update_bayesian_probabilities(new_data, treatment_success_counts):
    for treatment in ["Semaglutide", "Liraglutide", "Exenatide"]:
        success_count = len(new_data[(new_data["Treatment"] == treatment) & (new_data["MDS-UPDRS III Improvement"] < -4.0)])
        failure_count = len(new_data[(new_data["Treatment"] == treatment) & (new_data["MDS-UPDRS III Improvement"] >= -4.0)])
        treatment_success_counts[treatment]["success"] += success_count
        treatment_success_counts[treatment]["failures"] += failure_count

    updated_probs = {}
    for treatment in treatment_success_counts.keys():
        success = treatment_success_counts[treatment]["success"]
        failures = treatment_success_counts[treatment]["failures"]
        updated_probs[treatment] = beta.rvs(success + 1, failures + 1, size=5000).mean()
    
    return updated_probs

# Streamlit UI for AI-assisted Clinical Decision Tool
st.title("AI-Assisted Bayesian Treatment Selector for Parkinson's Disease")

# Patient Data Input
st.sidebar.header("Enter Patient Biomarker Data")

il6_reduction = st.sidebar.slider("IL-6 Reduction (%)", -20.0, 0.0, -13.0)
tnf_reduction = st.sidebar.slider("TNF-α Reduction (%)", -20.0, 0.0, -14.5)
dopamine_survival = st.sidebar.slider("Dopaminergic Neuron Survival (%)", 0.0, 10.0, 3.0)
mds_improvement = st.sidebar.slider("Expected MDS-UPDRS III Improvement", -10.0, 0.0, -4.5)

# Load current treatment probabilities
treatment_probs = load_treatment_probabilities()

# Bayesian Decision Model
best_treatment = max(treatment_probs, key=treatment_probs.get)

st.subheader("Recommended Treatment Based on Bayesian Decision Model")
st.write(f"**Optimal Treatment: {best_treatment}**")

st.subheader("Current Bayesian Treatment Probabilities")
st.table(pd.DataFrame(treatment_probs.items(), columns=["Treatment", "Probability of Being Best"]))

# Continuous Learning: Updating Model with New Patient Data
if st.sidebar.button("Update Model with New Patient Outcome"):
    new_patient_df = pd.DataFrame({
        "Treatment": [best_treatment],
        "IL-6 Reduction (%)": [il6_reduction],
        "TNF-α Reduction (%)": [tnf_reduction],
        "Dopaminergic Neuron Survival (%)": [dopamine_survival],
        "MDS-UPDRS III Improvement": [mds_improvement]
    })
    
    # Update probabilities dynamically
    updated_probs = update_bayesian_probabilities(new_patient_df, {
        "Semaglutide": {"success": 80, "failures": 20},
        "Liraglutide": {"success": 70, "failures": 30},
        "Exenatide": {"success": 60, "failures": 40}
    })
    
    st.subheader("Updated Bayesian Treatment Probabilities After New Patient Data")
    st.table(pd.DataFrame(updated_probs.items(), columns=["Treatment", "Updated Probability of Being Best"]))
