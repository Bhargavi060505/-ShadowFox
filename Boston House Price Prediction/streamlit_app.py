import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("house_model.pkl")

# Page config
st.set_page_config(page_title="Boston House Price Predictor", layout="centered")

# Title
st.title("🏠 Boston House Price Predictor")
st.markdown("Fill in the house features below to estimate its price (in $1000s):")

# Input fields in the main body
CRIM = st.text_input("Crime Rate (CRIM)", value="0.1")
ZN = st.text_input("Residential Land Zoned (ZN)", value="0.0")
INDUS = st.text_input("Non-Retail Business Acres (INDUS)", value="7.0")
CHAS = st.selectbox("Charles River Bounds (CHAS)", options=["No", "Yes"])
NOX = st.text_input("Nitric Oxide Concentration (NOX)", value="0.5")
RM = st.text_input("Average Number of Rooms (RM)", value="6.0")
AGE = st.text_input("Proportion of Older Buildings (AGE)", value="60")
DIS = st.text_input("Distance to Employment Centers (DIS)", value="4.0")
RAD = st.text_input("Accessibility to Highways (RAD)", value="1")
TAX = st.text_input("Property Tax Rate (TAX)", value="300")
PTRATIO = st.text_input("Pupil-Teacher Ratio (PTRATIO)", value="18.0")
B = st.text_input("Proportion of Black Population (B)", value="390.0")
LSTAT = st.text_input("Lower Status Population (%) (LSTAT)", value="12.0")

# Convert CHAS to numeric
chas_val = 1 if CHAS == "Yes" else 0

# Prediction
if st.button("Predict House Price"):
    try:
        input_features = np.array([[float(CRIM), float(ZN), float(INDUS), chas_val,
                                    float(NOX), float(RM), float(AGE), float(DIS),
                                    float(RAD), float(TAX), float(PTRATIO), float(B), float(LSTAT)]])
        
        prediction = model.predict(input_features)[0]
        st.success(f"🏡 Estimated House Price: ${prediction * 1000:,.2f}")
    
    except ValueError:
        st.error("❌ Please enter valid numeric values for all fields.")

# Footer
st.markdown("---")

