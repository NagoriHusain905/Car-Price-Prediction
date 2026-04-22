import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and columns
model = joblib.load('model/car_price_model.pkl')
model_columns = joblib.load('model/model_columns.pkl')

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to estimate the market price.")

# 2. Create Input Fields for the User
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Model Year", min_value=1990, max_value=2026, value=2015)
    engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0)

with col2:
    horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=150)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000)

# 3. The "Predict" Button
if st.button("Predict Price"):
    # Create a dataframe for the input
    input_df = pd.DataFrame([[year, engine, horsepower, mileage]], 
                            columns=['Model_Year', 'Engine_Size', 'Horsepower', 'Mileage'])
    
    # Fill in missing columns (for One-Hot Encoding dummy variables) with 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Make prediction
    prediction = model.predict(input_df)
    
    st.success(f"The estimated price for this car is: ${prediction[0]:,.2f}")
