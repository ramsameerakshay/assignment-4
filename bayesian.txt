# BAYESIAN NETWORK IMPLEMENTATION IN PYTHON
# Example:
# Medical Diagnosis using Bayesian Network
# Problem:
# Predict whether a patient has Flu based on:
# 1. Fever
# 2. Cough
# Bayesian Networks represent probabilistic
# relationships between variables.

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# CREATE BAYESIAN NETWORK STRUCTURE
# Flu -> Fever
# Flu -> Cough

model = BayesianNetwork([
    ('Flu', 'Fever'),
    ('Flu', 'Cough')
])

# DEFINE PROBABILITIES (CPD)
# Probability of Flu
cpd_flu = TabularCPD(
    variable='Flu',
    variable_card=2,
    values=[[0.95], [0.05]]
)

# Probability of Fever given Flu
cpd_fever = TabularCPD(
    variable='Fever',
    variable_card=2,
    values=[
        [0.90, 0.20],   # No Fever
        [0.10, 0.80]    # Fever
    ],
    evidence=['Flu'],
    evidence_card=[2]
)

# Probability of Cough given Flu
cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[
        [0.85, 0.30],   # No Cough
        [0.15, 0.70]    # Cough
    ],
    evidence=['Flu'],
    evidence_card=[2]
)

# ------------------------------------------
# ADD CPDs TO MODEL
# ------------------------------------------

model.add_cpds(cpd_flu, cpd_fever, cpd_cough)

# Verify correctness
print("Model Validity:", model.check_model())

# ------------------------------------------
# PERFORM INFERENCE
# ------------------------------------------

inference = VariableElimination(model)

# Query:
# Probability of Flu given Fever and Cough

result = inference.query(
    variables=['Flu'],
    evidence={
        'Fever': 1,
        'Cough': 1
    }
)

# ------------------------------------------
# DISPLAY RESULT
# ------------------------------------------

print("\n===== INFERENCE RESULT =====")
print("Probability of Flu given Fever and Cough:\n")

print(result)
