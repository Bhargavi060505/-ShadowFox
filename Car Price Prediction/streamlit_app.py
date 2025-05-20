import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("car_price_model.pkl")

# App title
st.title("🚗 Car Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Car Details:")

# User inputs
present_price = st.sidebar.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission Type", ['Manual', 'Automatic'])
car_age = st.sidebar.slider("Car Age (Years)", min_value=0, max_value=30, step=1)

# One-hot encoding for categorical features
fuel_petrol = 1 if fuel_type == 'Petrol' else 0
fuel_diesel = 1 if fuel_type == 'Diesel' else 0
seller_individual = 1 if seller_type == 'Individual' else 0
trans_manual = 1 if transmission == 'Manual' else 0

# Predict button
if st.sidebar.button("Predict Price"):
    features = np.array([[present_price, kms_driven, owner, car_age,
                          fuel_diesel, fuel_petrol, seller_individual, trans_manual]])
    
    prediction = model.predict(features)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

# Footer
st.markdown("---")
st.markdown("Thank you..!!")
