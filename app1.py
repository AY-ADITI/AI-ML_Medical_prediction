import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("model1.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.title("Diabetes Prediction App 🩺")

st.write("Enter the following medical details to check for diabetes risk.")

# Creating input fields for the 8 features in the dataset
feature_names = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
]

features = []
for name in feature_names:
    features.append(st.number_input(f"{name}", value=0.0))

# Predict button
if st.button("Predict"):
    input_data = np.array([features]).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 The person is **Diabetic**.")
    else:
        st.success("🟢 The person is **Not Diabetic**.")
