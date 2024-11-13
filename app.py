import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the trained model 
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scalers
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


##STREAMLIT APP

# Title and subtitle
st.title("🔍 Customer Churn Prediction")
st.write("Predict whether a customer will churn based on their profile and account information. Fill in the details below:")

# Adding custom styling and section headers
st.markdown("---")
st.subheader("📝 Customer Details")

# User input with sections and tooltips
with st.container():
    geography = st.selectbox('🌍 Geography', onehotencoder_geo.categories_[0], help="Select the customer's country.")
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_, help="Select the customer's gender.")
    age = st.slider('🎂 Age', 18, 92, help="Select the customer's age.")

st.markdown("---")
st.subheader("💳 Financial Details")

with st.container():
    credit_score = st.number_input('📈 Credit Score', min_value=300, max_value=850, help="Enter the customer's credit score.")
    balance = st.number_input('💰 Balance', min_value=0.0, help="Enter the customer's current account balance.")
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, help="Enter the customer's estimated annual salary.")

st.markdown("---")
st.subheader("🏦 Account Details")

with st.container():
    tenure = st.slider('📅 Tenure (Years)', 0, 10, help="Select the number of years the customer has been with the bank.")
    num_of_products = st.slider('📦 Number of Products', 1, 4, help="Select the number of products the customer has with the bank.")
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Does the customer have a credit card?")
    is_active_member = st.selectbox('📈 Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="Is the customer an active member?")

st.markdown("---")


# Additional styling and message
st.markdown("""
<style>
    .reportview-container .markdown-text-container {
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color:white;
        font-size:18px;
        border-radius:5px;
    }
</style>
""", unsafe_allow_html=True)


#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba>=0.5:
    st.write('The Customer is likely to Churn')
else:
    st.write('The Customer is not likely to Churn')