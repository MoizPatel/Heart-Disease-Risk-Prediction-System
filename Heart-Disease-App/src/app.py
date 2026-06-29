import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and columns
model = joblib.load('../model/heart_disease_model.pkl')
model_columns = joblib.load('../data/model_columns.pkl')

# 2. Build the Web Interface Title
st.title("Heart Disease Risk Prediction System")
st.write("Enter the patient's medical details below to calculate their risk.")

# 3. Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina?", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# 4. The Prediction Button
if st.button("Predict Risk"):
    # Store the inputs in a dictionary
    patient_data = {
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # One-Hot Encode
    patient_encoded = pd.get_dummies(patient_df)
    
    # Align columns with the training data
    patient_encoded = patient_encoded.reindex(columns=model_columns, fill_value=False)
    
    # Make prediction
    prediction = model.predict(patient_encoded)[0]
    probabilities = model.predict_proba(patient_encoded)[0]
    risk_confidence = probabilities[1] * 100
    
    # Display results
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **DIAGNOSIS: HIGH RISK** (Confidence: {risk_confidence:.1f}%)")
        st.write("Please consult a cardiologist immediately.")
    else:
        st.success(f"✅ **DIAGNOSIS: LOW RISK** (Confidence: {100 - risk_confidence:.1f}%)")
        st.write("Patient vitals look good!")