import streamlit as st
import pandas as pd
import joblib


model = joblib.load('alzheimers_model.pk1')

train_columns = ['PatientID', 'Age', 'BMI', 'Smoking', 'AlcoholConsumption',
       'PhysicalActivity', 'DietQuality', 'SleepQuality', 'SystolicBP',
       'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
       'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
       'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Gender_Male',
       'Ethnicity_Asian', 'Ethnicity_Caucasian', 'Ethnicity_Other',
       'EducationLevel_High School', 'EducationLevel_Higher',
       'EducationLevel_None', 'FamilyHistoryAlzheimers_Yes',
       'CardiovascularDisease_Yes', 'Diabetes_Yes', 'Depression_Yes',
       'HeadInjury_Yes', 'Hypertension_Yes', 'Confusion_Yes',
       'Disorientation_Yes', 'PersonalityChanges_Yes',
       'DifficultyCompletingTasks_Yes', 'Forgetfulness_Yes']

def get_patients_info():
    age = st.number_input('Age', min_value=0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
    education_level = st.selectbox('Education Level', ['None', 'High School', "Bachelor's", 'Higher'])
    bmi = st.number_input('BMI')
    smoking = st.selectbox('Smoking', ['No', 'Yes'])
    alcohol_consumption = st.number_input('Alcohol Consumption')
    physical_activity = st.number_input('Physical Activity')
    diet_quality = st.number_input("Diet Quality")
    sleep_quality = st.number_input('Sleep Quality')
    family_history = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'])
    cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    depression = st.selectbox('Depression', ['No', 'Yes'])
    head_injury = st.selectbox('Head Injury', ['No', 'Yes'])
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    systolicBP = st.number_input('Systolic BP')
    diastolicBP = st.number_input('diastolic BP')
    cholesterol_total = st.number_input('Cholesterol Total')
    CholesterolLDL = st.number_input('Cholesterol LDL')
    CholesterolHDL = st.number_input('Cholesterol HDL')
    CholesterolTriglyceride  = st.number_input('Cholesterol Triglyceride')
    mmse = st.number_input('MMSE')
    FunctionalAssessment = st.number_input('Functional Assessment')
    MemoryComplaints = st.selectbox('Memory Complaints', ['No', 'Yes'])
    BehavioralProblems = st.selectbox('Behavioral Problems', ['No', 'Yes'])
    adl = st.number_input('ADL')
    confusion = st.selectbox('Confusion', ['No', 'Yes'])
    disorientation = st.selectbox('Disorientation', ['No', 'Yes'])
    personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'])
    difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'])
    forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'])

    # Convert inputs to numeric for prediction
    gender = 0 if gender == 'Male' else 1
    ethnicity = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}[ethnicity]
    education_level = {'None': 0, 'High School': 1, "Bachelor's": 2, 'Higher': 3}[education_level]
    alcohol_consumption = 0 if alcohol_consumption == 'No' else 1
    family_history = 0 if family_history == 'No' else 1
    cardiovascular_disease = 0 if cardiovascular_disease == 'No' else 1
    diabetes = 0 if diabetes == 'No' else 1
    depression = 0 if depression == 'No' else 1
    head_injury = 0 if head_injury == 'No' else 1
    hypertension = 0 if hypertension == 'No' else 1
    MemoryComplaints = 0 if MemoryComplaints == 'No' else 1
    BehavioralProblems = 0 if BehavioralProblems == 'No' else 1
    confusion = 0 if confusion == 'No' else 1
    disorientation = 0 if disorientation == 'No' else 1
    personality_changes = 0 if personality_changes == 'No' else 1
    difficulty_completing_tasks = 0 if difficulty_completing_tasks == 'No' else 1
    forgetfulness = 0 if forgetfulness == 'No' else 1

    # Create the input data frame
    input_data = pd.DataFrame([[age, gender, ethnicity, education_level, bmi, smoking, alcohol_consumption, physical_activity, diet_quality, sleep_quality, family_history, cardiovascular_disease,
                                diabetes, depression, head_injury, hypertension,
                                systolicBP, diastolicBP, cholesterol_total, CholesterolLDL, CholesterolHDL, CholesterolTriglyceride, mmse, FunctionalAssessment,
                                MemoryComplaints, BehavioralProblems, adl, confusion, disorientation,
                                personality_changes, difficulty_completing_tasks, forgetfulness]],
                                columns=['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
                                        'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
                                        'SleepQuality', 'FamilyHistoryAlzheimers_Yes', 'CardiovascularDisease_Yes',
                                        'Diabetes_Yes', 'Depression_Yes', 'HeadInjury_Yes', 'Hypertension_Yes', 'SystolicBP',
                                        'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                                        'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
                                        'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion_Yes',
                                        'Disorientation_Yes', 'PersonalityChanges_Yes', 'DifficultyCompletingTasks_Yes',
                                        'Forgetfulness_Yes', ])

    # Convert categorical features into dummy variables (just like the model was trained on)
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align columns: add missing columns if necessary, and fill with 0
    missing_cols = set(train_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Ensure the order of columns is the same
    input_data = input_data[train_columns]


    # Use the model to predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Probability of Alzheimer's diagnosis (Yes)

    # Store the prediction and probability in session_state
    st.session_state.prediction = prediction[0]
    st.session_state.prediction_proba = prediction_proba[0]

    if prediction == 1:
        st.write(f"Prediction: Alzheimer's diagnosis likely. Probability: {prediction_proba[0]:.2f}")
    else:
        st.write(f"Prediction: No Alzheimer's diagnosis. Probability: {prediction_proba[0]:.2f}")

# UI setup
st.title("Alzheimer's Diagnosis Prediction")
st.write("Please provide the following details about the patient:")

if st.button('Get Prediction'):
    get_patients_info()