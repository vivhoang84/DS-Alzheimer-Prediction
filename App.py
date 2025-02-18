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
        "age": 0, "gender": "Enter Answer", "ethnicity": "Enter Answer", "education_level": "Enter Answer",
        "bmi": 0.0, "smoking": "Enter Answer", "alcohol_consumption": 0.0, "physical_activity": 0.0,
        "diet_quality": 0.0, "sleep_quality": 0.0, "family_history": "Enter Answer",
        "cardiovascular_disease": "Enter Answer", "diabetes": "Enter Answer", "depression": "Enter Answer",
        "head_injury": "Enter Answer", "hypertension": "Enter Answer", "systolicBP": 0.0,
        "diastolicBP": 0.0, "cholesterol_total": 0.0, "CholesterolLDL": 0.0,
        "CholesterolHDL": 0.0, "CholesterolTriglyceride": 0.0, "mmse": 0.0,
        "FunctionalAssessment": 0.0, "MemoryComplaints": "Enter Answer", "BehavioralProblems": "Enter Answer",
        "adl": 0.0, "confusion": "Enter Answer", "disorientation": "Enter Answer", "personality_changes": "Enter Answer",
        "difficulty_completing_tasks": "Enter Answer", "forgetfulness": "Enter Answer"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_patients_info():

    # age
    st.session_state.age = st.number_input('Enter your age', min_value=0, value=st.session_state.age)
    
    # gender
    gender_options = ['Enter Answer', 'Male', 'Female']
    gender_value = st.session_state.gender if st.session_state.gender in gender_options else 'Enter Answer'
    st.session_state.gender = st.selectbox(
        'Gender', gender_options, index=gender_options.index(gender_value)
    )

    # ethnicity
    ethnicity_options = ['Enter Answer', 'Caucasian', 'African American', 'Asian', 'Other']
    ethnicity_value = st.session_state.ethnicity if st.session_state.ethnicity in ethnicity_options else 'Enter Answer'
    st.session_state.ethnicity = st.selectbox(
        'Ethnicity', ethnicity_options, index=ethnicity_options.index(ethnicity_value)
    )

    # education level
    education_options = ['Enter Answer', 'None', 'High School', "Bachelor's", 'Higher']
    education_value = st.session_state.education_level if st.session_state.education_level in education_options else 'Enter Answer'
    st.session_state.education_level = st.selectbox(
        'Highest Education Level', education_options, index=education_options.index(education_value)
    )

    # BMI
    st.session_state.bmi = st.number_input('BMI (kg/m^2)', value=st.session_state.bmi)

    # smoking
    smoking_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.smoking = st.selectbox('Do you smoke?', smoking_options, index=smoking_options.index(st.session_state.smoking))

    # alcohol consumption
    st.session_state.alcohol_consumption = st.number_input('Weekly Alcohol Consumption', value=st.session_state.alcohol_consumption)

    # physical activity
    st.session_state.physical_activity = st.number_input('Weekly Physical Activity (hours)', value=st.session_state.physical_activity)
    
    # diet quality
    st.session_state.diet_quality = st.number_input("Diet Quality Score", value=st.session_state.diet_quality)
    
    # sleep quality
    st.session_state.sleep_quality = st.number_input('Sleep Quality Score', value=st.session_state.sleep_quality)
    
    # family history
    family_history_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.family_history = st.selectbox("Does anyone in your family have Alzheimer's?", family_history_options, index=family_history_options.index(st.session_state.family_history))
    
    # cardiovascular disease
    cardiovascular_disease_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.cardiovascular_disease = st.selectbox("Do you have cardiovascular disease?", cardiovascular_disease_options, index=cardiovascular_disease_options.index(st.session_state.cardiovascular_disease))
    
    # diabetes
    diabetes_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.diabetes = st.selectbox("Do you have diabetes?", diabetes_options, index=diabetes_options.index(st.session_state.diabetes))


    # head injuries
    head_injury_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.head_injury = st.selectbox("Have you had any head injuries?", head_injury_options, index=head_injury_options.index(st.session_state.head_injury))

    # hypertension
    hypertension_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.hypertension = st.selectbox("Do you have hypertension?", hypertension_options, index=hypertension_options.index(st.session_state.hypertension))


    st.session_state.systolicBP = st.number_input('Systolic BP', value=st.session_state.systolicBP)
    st.session_state.diastolicBP = st.number_input('Diastolic BP', value=st.session_state.diastolicBP)
    st.session_state.cholesterol_total = st.number_input('Cholesterol Total', value=st.session_state.cholesterol_total)
    st.session_state.CholesterolLDL = st.number_input('Cholesterol LDL', value=st.session_state.CholesterolLDL)
    st.session_state.CholesterolHDL = st.number_input('Cholesterol HDL', value=st.session_state.CholesterolHDL)
    st.session_state.CholesterolTriglyceride = st.number_input('Cholesterol Triglyceride', value=st.session_state.CholesterolTriglyceride)
    st.session_state.mmse = st.number_input('MMSE', value=st.session_state.mmse)
    st.session_state.FunctionalAssessment = st.number_input('Functional Assessment', value=st.session_state.FunctionalAssessment)

    # memory complaints
    MemoryComplaints_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.MemoryComplaints = st.selectbox("Do you have memory complaints?", MemoryComplaints_options, index=MemoryComplaints_options.index(st.session_state.MemoryComplaints))

    # behavioral problems
    BehavioralProblems_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.BehavioralProblems = st.selectbox("Do you have cardiovasculardisease?", BehavioralProblems_options, index=BehavioralProblems_options.index(st.session_state.BehavioralProblems))


    st.session_state.adl = st.number_input('ADL', value=st.session_state.adl)

    # confusion
    confusion_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.confusion = st.selectbox("Do you have confusion?", confusion_options, index=confusion_options.index(st.session_state.confusion))


    # disorientation
    disorientation_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.disorientation = st.selectbox("Do you have disorientation?", disorientation_options, index=disorientation_options.index(st.session_state.disorientation))


    # personality changes
    personality_changes_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.personality_changes = st.selectbox("Do you have personality changes?", personality_changes_options, index=personality_changes_options.index(st.session_state.personality_changes))

    # difficulty completing tasks
    difficulty_completing_tasks_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.difficulty_completing_tasks = st.selectbox("Do you have a difficult time completing tasks?", difficulty_completing_tasks_options, index=difficulty_completing_tasks_options.index(st.session_state.difficulty_completing_tasks))

    # forgetfulness
    forgetfulness_options = ['Enter Answer', 'No', 'Yes']
    st.session_state.forgetfulness = st.selectbox("Do you often expreience forgetfulness?", forgetfulness_options, index=forgetfulness_options.index(st.session_state.forgetfulness))


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
        st.write(f"\nPrediction: Alzheimer's diagnosis likely\nProbability: {prediction_proba[0]:.2f}")
    else:
        st.write(f"\nPrediction: No Alzheimer's diagnosis\nProbability: {prediction_proba[0]:.2f}")

# UI setup
st.title("Alzheimer's Diagnosis Prediction")
st.write("Please provide the following details about the patient:")

# Initialize session state
init_session_state()

# Display input fields
get_patients_info()

if st.button('Get Prediction'):
    predict()