import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Custom CSS to change font to Times New Roman
st.markdown("""
    <style>
    * {
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data and create dataframe
df = pd.read_csv("IT_customer_churn.csv")

# Convert TotalCharges to numeric and remove rows with spaces
df1 = df[df["TotalCharges"] != " "]
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

# Replace 'No internet service' and 'No phone service' with 'No'
df1.replace("No internet service", "No", inplace=True)
df1.replace("No phone service", "No", inplace=True)

# Convert Yes/No to 1/0
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# One hot encoding for categorical columns
df1['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
df1 = pd.get_dummies(df1, ['Contract', 'PaymentMethod', 'InternetService'], drop_first=True)

# Scaling
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

# Train-test split
X = df1.drop('Churn', axis='columns')
y = df1.Churn.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

# XGBoost model function that returns the trained model
def xgboost_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Prediction function
def predict_churn(user_input):
    # Ensure correct column order and align with X_train columns
    user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

    # Apply scaling to the relevant columns
    cols_to_scale_existing = [col for col in cols_to_scale if col in user_input.columns]
    if cols_to_scale_existing:
        user_input[cols_to_scale_existing] = scaler.transform(user_input[cols_to_scale_existing])

    # Train XGBoost model
    model = xgboost_model(X_train, y_train)

    # Predict using the trained model
    prediction = model.predict(user_input)
    return prediction

# Streamlit app UI
st.title("Customer Churn Prediction")

# Input options for the user
gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, step=1)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])

Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0, max_value=150, step=1)
TotalCharges = st.number_input("Total Charges", min_value=0, max_value=10000, step=1)

# Convert user inputs to DataFrame
user_input = pd.DataFrame({
    'gender': [1 if gender == 'Male' else 0],
    'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
    'Partner': [1 if Partner == 'Yes' else 0],
    'Dependents': [1 if Dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if PhoneService == 'Yes' else 0],
    'MultipleLines': [1 if MultipleLines == 'Yes' else 0],
    'OnlineSecurity': [1 if OnlineSecurity == 'Yes' else 0],
    'OnlineBackup': [1 if OnlineBackup == 'Yes' else 0],
    'DeviceProtection': [1 if DeviceProtection == 'Yes' else 0],
    'TechSupport': [1 if TechSupport == 'Yes' else 0],
    'StreamingTV': [1 if StreamingTV == 'Yes' else 0],
    'StreamingMovies': [1 if StreamingMovies == 'Yes' else 0],
    'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0],
    'Contract_One year': [1 if Contract == 'One year' else 0],
    'Contract_Two year': [1 if Contract == 'Two year' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if PaymentMethod == 'Credit card (automatic)' else 0],
    'PaymentMethod_Electronic check': [1 if PaymentMethod == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if PaymentMethod == 'Mailed check' else 0],
    'InternetService_Fiber optic': [1 if InternetService == 'Fiber optic' else 0],
    'InternetService_No': [1 if InternetService == 'No' else 0],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# Button for prediction
if st.button("Predict Churn"):
    prediction = predict_churn(user_input)
    if prediction == 1:
        st.write("The customer is predicted to churn.")
    else:
        st.write("The customer is predicted to not churn.")
