import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle

# -------------------------
# Load trained model
# -------------------------

model = pickle.load(open("models/xgboost_model.pkl", "rb"))
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.set_page_config(
    page_title="NYC Family House Price Predictor",
    layout="centered"
)

st.title(" NYC Family House Price Predictor")
st.write("Estimate sale price for 1 to 3 family residential properties")

# -------------------------
# User Inputs
# -------------------------
st.subheader("Property Details")

borough = st.selectbox(
    "Borough",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Manhattan",
        2: "Bronx",
        3: "Brooklyn",
        4: "Queens",
        5: "Staten Island"
    }[x]
)

zip_code = st.number_input("ZIP Code", min_value=10001, max_value=11697, step=1)

res_units = st.number_input("Residential Units", min_value=1, max_value=3, step=1)
comm_units = st.number_input("Commercial Units", min_value=0, max_value=1, step=1)
total_units = res_units + comm_units

land_sqft = st.number_input("Land Square Feet", min_value=500, step=50)
gross_sqft = st.number_input("Gross Square Feet", min_value=500, step=50)

year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)

latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

# -------------------------
# Predict
# -------------------------
if st.button("Predict Price"):
    
    input_data = pd.DataFrame([{
        "BOROUGH": borough,
        "ZIP CODE": zip_code,
        "RESIDENTIAL UNITS": res_units,
        "COMMERCIAL UNITS": comm_units,
        "TOTAL UNITS": total_units,
        "LAND SQUARE FEET": land_sqft,
        "GROSS SQUARE FEET": gross_sqft,
        "YEAR BUILT": year_built,
        "latitude": latitude,
        "longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    st.success(f" Estimated Property Price: ${prediction:,.0f}")
