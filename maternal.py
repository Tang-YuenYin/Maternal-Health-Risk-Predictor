import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from dotenv import load_dotenv
import os
import toml
from sklearn.model_selection import train_test_split


def initialize_firebase():
    # Load Firebase credentials from Streamlit secrets
    firebase_credentials = st.secrets["firebase"]

    # Initialize Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": firebase_credentials["type"],
            "project_id": firebase_credentials["project_id"],
            "private_key_id": firebase_credentials["private_key_id"],
            "private_key": firebase_credentials["private_key"],
            "client_email": firebase_credentials["client_email"],
            "client_id": firebase_credentials["client_id"],
            "auth_uri": firebase_credentials["auth_uri"],
            "token_uri": firebase_credentials["token_uri"],
            "auth_provider_x509_cert_url": firebase_credentials["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_credentials["client_x509_cert_url"],
        })
        firebase_admin.initialize_app(cred)

    return firestore.client()

db = initialize_firebase()

# Load the data
@st.cache_data  # Add caching for better performance
def load_data():
    data = pd.read_csv('Maternal Health Risk Data Set.csv')
    return data

data = load_data()

# Label encode the target variable
le = LabelEncoder()
data['RiskLevel'] = le.fit_transform(data['RiskLevel'])

# Define the column order
feature_names = ['Age', 'BS', 'BodyTemp', 'DiastolicBP', 'HeartRate', 'SystolicBP']

# Create sidebar
st.sidebar.subheader('Data Exploration')
section = st.sidebar.selectbox('Maternal Health Risk', ('Data Rows', 'RiskLevel Counts', 'Data Description', 'RiskLevel Prediction'))

# Display selected section
if section == 'Data Rows':
    st.subheader("First 10 rows of the data:")
    st.write(data.head(10))
elif section == 'RiskLevel Counts':
    st.subheader("Counts of RiskLevel for each Age:")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Age', hue='RiskLevel', data=data)
    plt.title('Counts of RiskLevel for each Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot(plt)
elif section == 'Data Description':
    st.subheader("Description of Data:")
    st.write(data.describe().T)
elif section == 'RiskLevel Prediction':
    st.subheader("Predict RiskLevel")
    age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
    bs = st.number_input("Enter Blood Sugar", min_value=0.0, max_value=300.0, value=100.0)
    body_temp = st.number_input("Enter Body Temperature", min_value=35.0, max_value=42.0, value=37.0)
    diastolic_bp = st.number_input("Enter Diastolic Blood Pressure", min_value=0, max_value=200, value=80)
    heart_rate = st.number_input("Enter Heart Rate", min_value=0, max_value=200, value=70)
    systolic_bp = st.number_input("Enter Systolic Blood Pressure", min_value=0, max_value=300, value=120)

    input_data = pd.DataFrame({
        'Age': [age],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'DiastolicBP': [diastolic_bp],
        'HeartRate': [heart_rate],
        'SystolicBP': [systolic_bp]
    }, columns=feature_names)

    # Create a classification model
    X = data[feature_names]
    y = data['RiskLevel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    # prediction = model.predict(input_data)  # Make a prediction using the trained model

    # predicted_label = le.inverse_transform(prediction)[0]
    # st.write("Predicted Risk Level:", predicted_label)

    # if st.button("Save to Mama's Journal"):
    #         # Convert input_data DataFrame to dictionary
    #         input_data_dict = input_data.astype(float).to_dict(orient='records')[0]
    #         # Convert prediction value to a standard Python integer
    #         prediction_value = int(prediction[0])
    #         # Save prediction to Firebase
    #         doc_ref = db.collection("Maternal").add({
    #         "date": datetime.now().isoformat(),
    #         "input_data": input_data_dict ,
    #         "prediction": predicted_label ,
    #         "prediction_value": prediction_value,
    #     })
    #         st.success("Prediction saved to Firebase")
    if st.button("Predict"):
        input_data_dict = input_data.astype(float).to_dict(orient='records')[0]
        prediction = model.predict(input_data)  # Make a prediction using the trained model
        predicted_label = le.inverse_transform(prediction)[0]
        st.write("Predicted Risk Level:", predicted_label)
        
        # Store the prediction result in session state
        st.session_state["prediction_result"] = {
            "input_data": input_data_dict,
            "prediction_label": predicted_label,
            "prediction_value": int(prediction[0])
        }

# Check if there is a prediction result in session state and display the save button
if "prediction_result" in st.session_state:
    if st.button("Save to Mama's Journal"):
        result = st.session_state["prediction_result"]
        # Save prediction to Firebase
        doc_ref = db.collection("Maternal").add({
            "date": datetime.now().isoformat(),
            "input_data": result["input_data"],
            "prediction": result["prediction_label"],
            "prediction_value": result["prediction_value"]
        })
        st.success("Prediction saved to Firebase")
