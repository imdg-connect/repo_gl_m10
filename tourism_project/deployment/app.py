import streamlit as st
import pandas as pd
import joblib

model = joblib.load("tourism_project/deployment/best_model.joblib")

st.title("Wellness Tourism Package — Purchase Prediction")
st.write("Enter customer details to predict likelihood of purchase.")

age = st.number_input("Age", 18, 100, 35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
# "Unmarried" values were consolidated into "Single" during prep.py — don't add it here,
# the trained model never saw that category.
num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Holds Passport", [0, 1])
own_car = st.selectbox("Owns Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 1000, 100000, 20000)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Followups", 0, 10, 2)
duration_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": num_persons, "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
        "Passport": passport, "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children, "Designation": designation,
        "MonthlyIncome": monthly_income, "PitchSatisfactionScore": pitch_score,
        "ProductPitched": product_pitched, "NumberOfFollowups": num_followups,
        "DurationOfPitch": duration_pitch,
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to purchase (probability: {proba:.2%})")
    else:
        st.warning(f"❌ Unlikely to purchase (probability: {proba:.2%})")
