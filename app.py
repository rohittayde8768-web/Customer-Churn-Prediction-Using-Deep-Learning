import pandas as pd
import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the trained model

model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = joblib.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)
    
## Streamlit App
st.title("Customer Churn Prediction")

# User input for the features
geography = st.selectbox('Geography', one_hot_encoder.categories_[0],)

gender= st.selectbox('Gender', label_encoder.classes_)

age= st.selectbox('Age', options=list(range(19, 100)), index=10)

tenure= st.number_input('Tenure', min_value=0, max_value=10, value=3)

balance= st.number_input('Balance', )

credit_score= st.number_input('Credit Score')

num_of_products= st.number_input('Number of Products', min_value=1, max_value=4, value=1)

has_cr_card= st.selectbox('Has Credit Card', ['Yes', 'No'])

is_active_member= st.selectbox('Is Active Member', ['Yes', 'No'])

estimated_salary= st.number_input('Estimated Salary', )


## Prepare the input data 
input_data=pd.DataFrame({ 
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary':[estimated_salary], 
    'CreditScore':[credit_score]
})

## One hot encoding for 'Geography' column
geography_encoded = one_hot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

## Combine one hot encoded columns with the input data
#input_data=pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

#Scaling the input data
#input_scaled = scaler.transform(input_data)

# Combine one-hot encoded columns with input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geography_encoded_df],
    axis=1
)

# Arrange columns according to training order
input_data = input_data[scaler.feature_names_in_]

# Scale input data
input_scaled = scaler.transform(input_data)

##Predict the Churn
Prediction = model.predict(input_scaled)
Prediction_probability = Prediction[0][0]

st.write(f"Churn Probability: {Prediction_probability:.2f}")

if Prediction_probability > 0.5:
    st.write("The customer is likely to churn.")    
else:
    st.write("The customer is not likely to churn.")
    
# PREDICTION BUTTON
predict = st.button(
    "🔍  Verify Customer",
    use_container_width=True
)