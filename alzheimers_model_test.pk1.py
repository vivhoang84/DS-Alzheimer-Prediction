import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Loading Dataset
Alzheimer = "alzheimers_disease_data.csv"
alz = pd.read_csv(Alzheimer)

# Displaying the first few rows of the dataset
alz.head()

# Overview of Dataset
alz.info()

# Identifying Datatypes
alz.dtypes

# Mappings for categorical columns
replace_mappings = {
    'Gender': {0: 'Male', 1: 'Female'},
    'Ethnicity': {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'},
    'EducationLevel': {0: 'None', 1: 'High School', 2: "Bachelor's", 3: 'Higher'},
    'FamilyHistoryAlzheimers': {0: 'No', 1: 'Yes'},
    'CardiovascularDisease': {0: 'No', 1: 'Yes'},
    'Diabetes': {0: 'No', 1: 'Yes'},
    'Depression': {0: 'No', 1: 'Yes'},
    'HeadInjury': {0: 'No', 1: 'Yes'},
    'Hypertension': {0: 'No', 1: 'Yes'},
    'Confusion': {0: 'No', 1: 'Yes'},
    'Disorientation': {0: 'No', 1: 'Yes'},
    'PersonalityChanges': {0: 'No', 1: 'Yes'},
    'DifficultyCompletingTasks': {0: 'No', 1: 'Yes'},
    'Forgetfulness': {0: 'No', 1: 'Yes'},
    'Diagnosis': {0: 'No', 1: 'Yes'}
}

# Apply the replacements
for column, mapping in replace_mappings.items():
    if alz[column].dtype in ['int64', 'float64']:  # Ensure column is numeric before replacing
        alz[column] = alz[column].replace(mapping)

# Checking for NULL values
print("Null values: ", alz.isnull().sum().sum())

# Checking for duplicates
print("Duplicates: ", alz.duplicated().sum())

# Visualizations Setup
fig, axes = plt.subplots(7, 2, figsize=(15, 30))  # Adjusted the number of subplots for all 14 categories
axes = axes.flatten()

# Plot information for Age, Gender, Ethnicity, Education Level, and other categorical features
plots_info = [
    (sns.histplot, "Age", "Age", "Age Distribution of Patients by Alzheimer’s Diagnosis"),
    (sns.countplot, "Gender", "Gender", "Alzheimer’s Diagnosis Distribution by Gender"),
    (sns.countplot, "Ethnicity", "Ethnicity", "Alzheimer’s Diagnosis Distribution by Ethnicity"),
    (sns.countplot, "EducationLevel", "Education Level", "Alzheimer’s Diagnosis Distribution by Education Level"),
    (sns.countplot, "FamilyHistoryAlzheimers", "Family History of Alzheimer’s", "Alzheimer’s Diagnosis by Family History"),
    (sns.countplot, "CardiovascularDisease", "Cardiovascular Disease", "Alzheimer’s Diagnosis by Cardiovascular Disease"),
    (sns.countplot, "Diabetes", "Diabetes", "Alzheimer’s Diagnosis by Diabetes"),
    (sns.countplot, "Depression", "Depression", "Alzheimer’s Diagnosis by Depression"),
    (sns.countplot, "HeadInjury", "Head Injury", "Alzheimer’s Diagnosis by Head Injury"),
    (sns.countplot, "Hypertension", "Hypertension", "Alzheimer’s Diagnosis by Hypertension"),
    (sns.countplot, "Confusion", "Confusion", "Alzheimer’s Diagnosis by Confusion"),
    (sns.countplot, "Disorientation", "Disorientation", "Alzheimer’s Diagnosis by Disorientation"),
    (sns.countplot, "PersonalityChanges", "Personality Changes", "Alzheimer’s Diagnosis by Personality Changes"),
    (sns.countplot, "DifficultyCompletingTasks", "Difficulty Completing Tasks", "Alzheimer’s Diagnosis by Difficulty Completing Tasks"),
    (sns.countplot, "Forgetfulness", "Forgetfulness", "Alzheimer’s Diagnosis by Forgetfulness")
]

# Splitting data into training and testing sets
train_alzheimer, test_alzheimer = train_test_split(alz, test_size=0.2, random_state=42)
print(train_alzheimer.shape)
print(test_alzheimer.shape)

# Function to add labels to countplots
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black')

# Adjust space between plots
plt.subplots_adjust(hspace=0.5)

# Create plots for all categories
for ax, (plot_func, column, xlabel, title) in zip(axes, plots_info):
    plot_func(data=train_alzheimer, x=column, hue="Diagnosis", ax=ax)
    if plot_func == sns.countplot:
        add_labels(ax)  # Add labels to countplots
    ax.set_xlabel(xlabel)
    ax.set_title(title)

# Heatmap Setup
# Select only numeric columns from the training dataset
numeric_data = train_alzheimer.select_dtypes(include=["int64", "float64"])

# Check if there are any numeric columns for heatmap
if not numeric_data.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), cmap="YlGnBu", annot=True, fmt=".2f")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("No numeric data available for the heatmap.")

# Model Training and Evaluation
# Preparing features and labels
X = alz.drop(columns=['Diagnosis'])  # Drop target variable
y = alz['Diagnosis']  # Target variable

# Encoding categorical variables with pd.get_dummies
X = pd.get_dummies(X, drop_first=True)

# Splitting data into train and test sets using the same split as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Align columns between train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# GradientBoostingClassifier model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predicting on test data
y_pred = gb_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Alzheimer’s Diagnosis Prediction")
plt.show()

# Save the model
joblib.dump(gb_model, 'alzheimers_model.pkl')

# Citation
'''
@misc{rabie_el_kharoua_2024,
title={Alzheimer's Disease Dataset},
url={https://www.kaggle.com/dsv/8668279},
DOI={10.34740/KAGGLE/DSV/8668279},
publisher={Kaggle},
author={Rabie El Kharoua},
year={2024}
}
'''
