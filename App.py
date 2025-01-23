import streamlit as st
import pandas as pd
import joblib


model = joblib.load('alzheimers_model.pk1')

def get_patients_info():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
    education_level = st.selectbox('Education Level', ['None', 'High School', "Bachelor's", 'Higher'])
    family_history = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'])
    cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    depression = st.selectbox('Depression', ['No', 'Yes'])
    head_injury = st.selectbox('Head Injury', ['No', 'Yes'])
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    confusion = st.selectbox('Confusion', ['No', 'Yes'])
    disorientation = st.selectbox('Disorientation', ['No', 'Yes'])
    personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'])
    difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'])
    forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'])

    # Convert inputs to numeric for prediction
    gender = 0 if gender == 'Male' else 1
    ethnicity = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}[ethnicity]
    education_level = {'None': 0, 'High School': 1, "Bachelor's": 2, 'Higher': 3}[education_level]
    family_history = 0 if family_history == 'No' else 1
    cardiovascular_disease = 0 if cardiovascular_disease == 'No' else 1
    diabetes = 0 if diabetes == 'No' else 1
    depression = 0 if depression == 'No' else 1
    head_injury = 0 if head_injury == 'No' else 1
    hypertension = 0 if hypertension == 'No' else 1
    confusion = 0 if confusion == 'No' else 1
    disorientation = 0 if disorientation == 'No' else 1
    personality_changes = 0 if personality_changes == 'No' else 1
    difficulty_completing_tasks = 0 if difficulty_completing_tasks == 'No' else 1
    forgetfulness = 0 if forgetfulness == 'No' else 1

    # Create the input data frame
    input_data = pd.DataFrame([[gender, ethnicity, education_level, family_history, cardiovascular_disease,
                                diabetes, depression, head_injury, hypertension, confusion, disorientation,
                                personality_changes, difficulty_completing_tasks, forgetfulness]],
                              columns=['Gender', 'Ethnicity', 'EducationLevel', 'FamilyHistoryAlzheimers',
                                       'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
                                       'Hypertension', 'Confusion', 'Disorientation', 'PersonalityChanges',
                                       'DifficultyCompletingTasks', 'Forgetfulness'])

    # Use the model to predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Probability of Alzheimer's diagnosis (Yes)

    if prediction == 1:
        st.write(f"Prediction: Alzheimer's diagnosis likely. Probability: {prediction_proba[0]:.2f}")
    else:
        st.write(f"Prediction: No Alzheimer's diagnosis. Probability: {prediction_proba[0]:.2f}")

# UI setup
st.title("Alzheimer's Diagnosis Prediction")
st.write("Please provide the following details about the patient:")

if st.button('Get Prediction'):
    get_patients_info