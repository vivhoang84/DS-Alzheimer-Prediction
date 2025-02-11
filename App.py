import streamlit as st
import pandas as pd
import joblib

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('alzheimers_model.pk1')

model = load_model()

# Define the training columns
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

# Initialize session state variables for inputs
def init_session_state():
    defaults = {
        "age": 0, "gender": "Male", "ethnicity": "Caucasian", "education_level": "None",
        "bmi": 0.0, "smoking": "No", "alcohol_consumption": 0.0, "physical_activity": 0.0,
        "diet_quality": 0.0, "sleep_quality": 0.0, "family_history": "No",
        "cardiovascular_disease": "No", "diabetes": "No", "depression": "No",
        "head_injury": "No", "hypertension": "No", "systolicBP": 0.0,
        "diastolicBP": 0.0, "cholesterol_total": 0.0, "CholesterolLDL": 0.0,
        "CholesterolHDL": 0.0, "CholesterolTriglyceride": 0.0, "mmse": 0.0,
        "FunctionalAssessment": 0.0, "MemoryComplaints": "No", "BehavioralProblems": "No",
        "adl": 0.0, "confusion": "No", "disorientation": "No", "personality_changes": "No",
        "difficulty_completing_tasks": "No", "forgetfulness": "No"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_patients_info():
    st.session_state.age = st.number_input('Age', min_value=0, value=st.session_state.age)
    st.session_state.gender = st.selectbox('Gender', ['Male', 'Female'], index=['Male', 'Female'].index(st.session_state.gender))
    st.session_state.ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'], index=['Caucasian', 'African American', 'Asian', 'Other'].index(st.session_state.ethnicity))
    st.session_state.education_level = st.selectbox('Education Level', ['None', 'High School', "Bachelor's", 'Higher'], index=['None', 'High School', "Bachelor's", 'Higher'].index(st.session_state.education_level))
    st.session_state.bmi = st.number_input('BMI', value=st.session_state.bmi)
    st.session_state.smoking = st.selectbox('Smoking', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.smoking))
    st.session_state.alcohol_consumption = st.number_input('Alcohol Consumption', value=st.session_state.alcohol_consumption)
    st.session_state.physical_activity = st.number_input('Physical Activity', value=st.session_state.physical_activity)
    st.session_state.diet_quality = st.number_input("Diet Quality", value=st.session_state.diet_quality)
    st.session_state.sleep_quality = st.number_input('Sleep Quality', value=st.session_state.sleep_quality)
    st.session_state.family_history = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.family_history))
    st.session_state.cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.cardiovascular_disease))
    st.session_state.diabetes = st.selectbox('Diabetes', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.diabetes))
    st.session_state.depression = st.selectbox('Depression', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.depression))
    st.session_state.head_injury = st.selectbox('Head Injury', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.head_injury))
    st.session_state.hypertension = st.selectbox('Hypertension', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.hypertension))
    st.session_state.systolicBP = st.number_input('Systolic BP', value=st.session_state.systolicBP)
    st.session_state.diastolicBP = st.number_input('Diastolic BP', value=st.session_state.diastolicBP)
    st.session_state.cholesterol_total = st.number_input('Cholesterol Total', value=st.session_state.cholesterol_total)
    st.session_state.CholesterolLDL = st.number_input('Cholesterol LDL', value=st.session_state.CholesterolLDL)
    st.session_state.CholesterolHDL = st.number_input('Cholesterol HDL', value=st.session_state.CholesterolHDL)
    st.session_state.CholesterolTriglyceride = st.number_input('Cholesterol Triglyceride', value=st.session_state.CholesterolTriglyceride)
    st.session_state.mmse = st.number_input('MMSE', value=st.session_state.mmse)
    st.session_state.FunctionalAssessment = st.number_input('Functional Assessment', value=st.session_state.FunctionalAssessment)
    st.session_state.MemoryComplaints = st.selectbox('Memory Complaints', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.MemoryComplaints))
    st.session_state.BehavioralProblems = st.selectbox('Behavioral Problems', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.BehavioralProblems))
    st.session_state.adl = st.number_input('ADL', value=st.session_state.adl)
    st.session_state.confusion = st.selectbox('Confusion', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.confusion))
    st.session_state.disorientation = st.selectbox('Disorientation', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.disorientation))
    st.session_state.personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.personality_changes))
    st.session_state.difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.difficulty_completing_tasks))
    st.session_state.forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.forgetfulness))

def predict():
    # Convert inputs to numeric for prediction
    input_data = pd.DataFrame({
        'Age': [st.session_state.age],
        'Gender': [0 if st.session_state.gender == 'Male' else 1],
        'Ethnicity': [{'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}[st.session_state.ethnicity]],
        'EducationLevel': [{'None': 0, 'High School': 1, "Bachelor's": 2, 'Higher': 3}[st.session_state.education_level]],
        'BMI': [st.session_state.bmi],
        'Smoking': [0 if st.session_state.smoking == 'No' else 1],
        'AlcoholConsumption': [st.session_state.alcohol_consumption],
        'PhysicalActivity': [st.session_state.physical_activity],
        'DietQuality': [st.session_state.diet_quality],
        'SleepQuality': [st.session_state.sleep_quality],
        'FamilyHistoryAlzheimers_Yes': [0 if st.session_state.family_history == 'No' else 1],
        'CardiovascularDisease_Yes': [0 if st.session_state.cardiovascular_disease == 'No' else 1],
        'Diabetes_Yes': [0 if st.session_state.diabetes == 'No' else 1],
        'Depression_Yes': [0 if st.session_state.depression == 'No' else 1],
        'HeadInjury_Yes': [0 if st.session_state.head_injury == 'No' else 1],
        'Hypertension_Yes': [0 if st.session_state.hypertension == 'No' else 1],
        'SystolicBP': [st.session_state.systolicBP],
        'DiastolicBP': [st.session_state.diastolicBP],
        'CholesterolTotal': [st.session_state.cholesterol_total],
        'CholesterolLDL': [st.session_state.CholesterolLDL],
        'CholesterolHDL': [st.session_state.CholesterolHDL],
        'CholesterolTriglycerides': [st.session_state.CholesterolTriglyceride],
        'MMSE': [st.session_state.mmse],
        'FunctionalAssessment': [st.session_state.FunctionalAssessment],
        'MemoryComplaints': [0 if st.session_state.MemoryComplaints == 'No' else 1],
        'BehavioralProblems': [0 if st.session_state.BehavioralProblems == 'No' else 1],
        'ADL': [st.session_state.adl],
        'Confusion_Yes': [0 if st.session_state.confusion == 'No' else 1],
        'Disorientation_Yes': [0 if st.session_state.disorientation == 'No' else 1],
        'PersonalityChanges_Yes': [0 if st.session_state.personality_changes == 'No' else 1],
        'DifficultyCompletingTasks_Yes': [0 if st.session_state.difficulty_completing_tasks == 'No' else 1],
        'Forgetfulness_Yes': [0 if st.session_state.forgetfulness == 'No' else 1],
    })

    # Convert categorical features into dummy variables
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

    if prediction[0] == 1:
        st.write(f"Prediction: Alzheimer's diagnosis likely. Probability: {prediction_proba[0]:.2f}")
    else:
        st.write(f"Prediction: No Alzheimer's diagnosis. Probability: {prediction_proba[0]:.2f}")

# UI setup
st.title("Alzheimer's Diagnosis Prediction")
st.write("Please provide the following details about the patient:")

# Initialize session state
init_session_state()

# Display input fields
get_patients_info()

if st.button('Get Prediction'):
    predict()
