# app.py
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib

# Custom CSS for light blue background and black text
st.markdown("""
<style>
    /* Light blue background */
    .stApp, .main {
        background-color: #e3f2fd;
    }

    /* Default text: black */
    .stApp, .stApp * {
        color: black !important;
    }

    /* Inputs: white background, black text */
    input, textarea, .stNumberInput input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    /* DROPDOWN FIX: Light purple background with black text */
    .stSelectbox > div > div,
    .stSelectbox select,
    .stSelectbox option,
    div[data-baseweb="select"],
    div[data-baseweb="select"] * {
        background-color: #e6c6f6 !important;  /* Light purple */
        color: black !important;
    }

    /* Dropdown menu popup */
    ul[role="listbox"] {
        background-color: #e6c6f6 !important;
        color: black !important;
    }

    /* Dropdown options */
    ul[role="listbox"] li {
        background-color: #e6c6f6 !important;
        color: black !important;
    }

    /* Hover effect for options */
    ul[role="listbox"] li:hover {
        background-color: #ce93d8 !important;  /* Slightly darker purple on hover */
        color: black !important;
    }
    

    /* +/- Buttons inside NumberInput */
    .stNumberInput button {
        background-color: #e6c6f6 !important;
        border: none !important;
    }

    .stNumberInput button span {
        color: white !important;
    }

    .stNumberInput button:hover {
        background-color: #c775f0 !important;
    }

    /* Predict button */
    .stButton button {
        background-color: #e6c6f6 !important;
        border: none !important;
    }

    .stButton button span {
        color: white !important;
    }

    .stButton button:hover {
        background-color: #c775f0 !important;
    }
</style>
""", unsafe_allow_html=True)




def calculate_diabetes_pedigree_function(parents_diabetes, siblings_diabetes, 
                                       grandparents_diabetes, aunts_uncles_diabetes,
                                       age_diagnosed_parent=None, age_diagnosed_sibling=None):

    
    # Base weights for different relative types (closer relatives have higher weight)
    weights = {
        'parents': 0.5,      # First degree relatives
        'siblings': 0.5,     # First degree relatives  
        'grandparents': 0.25,  # Second degree relatives
        'aunts_uncles': 0.125  # Second degree relatives
    }
    
    # Calculate weighted family history score
    family_score = 0
    
    # Parents contribution
    if parents_diabetes > 0:
        parent_contribution = weights['parents'] * parents_diabetes
        # Adjust for early diagnosis (younger age = higher risk)
        if age_diagnosed_parent and age_diagnosed_parent < 50:
            parent_contribution *= 1.5
        elif age_diagnosed_parent and age_diagnosed_parent < 40:
            parent_contribution *= 2.0
        family_score += parent_contribution
    
    # Siblings contribution
    if siblings_diabetes > 0:
        sibling_contribution = weights['siblings'] * siblings_diabetes
        # Adjust for early diagnosis
        if age_diagnosed_sibling and age_diagnosed_sibling < 40:
            sibling_contribution *= 1.5
        elif age_diagnosed_sibling and age_diagnosed_sibling < 30:
            sibling_contribution *= 2.0
        family_score += sibling_contribution
    
    # Grandparents and aunts/uncles
    family_score += weights['grandparents'] * grandparents_diabetes
    family_score += weights['aunts_uncles'] * aunts_uncles_diabetes
    
    # Convert to DPF scale (typically 0.0 to 2.5, with mean around 0.4-0.5)
    # Apply a logarithmic transformation to match typical DPF distribution
    if family_score == 0:
        estimated_dpf = 0.078  # Minimum baseline value from typical dataset
    else:
        # Scale and transform to match typical DPF range
        estimated_dpf = 0.078 + (family_score * 0.3) + np.log1p(family_score) * 0.2
        
        # Cap at reasonable maximum
        estimated_dpf = min(estimated_dpf, 2.42)  # Typical maximum from dataset
    
    return round(estimated_dpf, 3)




#loading model

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_scaler():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "DiabetesNN3.keras")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    skinthickness_model_path = os.path.join(current_dir, "skinthickness_model.pkl")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    skinthickness_model = joblib.load(skinthickness_model_path)

    return model, scaler, skinthickness_model


try:
    model, scaler, skin_thickness_model = load_model_and_scaler()
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    model = None
    scaler = None
    skin_thickness_model = None




st.title("Diabetes Prediction App for Female Population")

st.write("Enter patient data:")
st.write("Data can be entered using +/- buttons as well as by keyboard.")

# Get user input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (enter 0 if unknown)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
#dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if skin_thickness == 0 and skin_thickness_model:
    input_for_skin = np.array([[bmi, insulin, blood_pressure]])
    predicted_skin = skin_thickness_model.predict(input_for_skin)[0]
    skin_thickness = round(predicted_skin, 1)
    st.warning(f"Skin Thickness was missing. Estimated value: **{skin_thickness} mm**")

st.header("Family History of Diabetes")
st.write("Please answer these questions about your family's diabetes history:")

parents_diabetes = st.selectbox(
    "How many of your parents have/had diabetes?",
    options=[0, 1, 2],
    format_func=lambda x: f"{x} parent(s)" if x != 1 else "1 parent",
    help="Include both biological mother and father"
)

age_diagnosed_parent = None
if parents_diabetes > 0:
    age_diagnosed_parent = st.number_input(
        "Age when first parent was diagnosed (if known):",
        min_value=1, max_value=100, value=50,
        help="Leave as default if unknown"
    )

siblings_diabetes = st.number_input(
    "How many of your siblings have/had diabetes?",
    min_value=0, max_value=10, value=0,
    help="Include all biological brothers and sisters"
)

age_diagnosed_sibling = None
if siblings_diabetes > 0:
    age_diagnosed_sibling = st.number_input(
        "Age when first sibling was diagnosed (if known):",
        min_value=1, max_value=100, value=45,
        help="Leave as default if unknown"
    )

grandparents_diabetes = st.number_input(
    "How many of your grandparents had diabetes?",
    min_value=0, max_value=4, value=0,
    help="Include all 4 grandparents if known"
)

aunts_uncles_diabetes = st.number_input(
    "How many aunts/uncles have/had diabetes?",
    min_value=0, max_value=20, value=0,
    help="Approximate number if you're not sure of the exact count"
)

# Calculate and display the estimated DPF
dpf = calculate_diabetes_pedigree_function(
    parents_diabetes, siblings_diabetes, grandparents_diabetes, 
    aunts_uncles_diabetes, age_diagnosed_parent, age_diagnosed_sibling
)

st.info(f"**Calculated Family Risk Score:** {dpf}")




# Make prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)              # Since scaling was used during the training

    prediction = model.predict(input_scaled)[0][0]
    prediction_class = int(prediction > 0.5)

    st.subheader("Result:")
    if prediction_class == 1:
        st.error("The model predicts the person is **likely diabetic.**")
    else:
        st.success("The model predicts the person is **not diabetic.**")

    st.write(f"Prediction probability: **{prediction:.4f}**")



st.markdown("---")
st.subheader("Model Accuracy Overview")

st.markdown("""
**Diabetes Prediction**
- Accuracy: **77%**  
_This model correctly predicts diabetes in about 77 out of 100 cases._

** Skin Thickness Estimation**
- Prediction Confidence: **Moderate (39%)**  
_This model estimates missing skin thickness values using BMI, insulin, and blood pressure._
""")

st.markdown("---")  # horizontal line to separate
st.markdown(
    "<p style='text-align:center; color: gray; font-size: 12px;'>"
    "Prediction model trained on Pima Indian Diabetes dataset"
    "</p>",
    unsafe_allow_html=True
)