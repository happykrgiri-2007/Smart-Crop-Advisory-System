
import streamlit as st

st.set_page_config(
    page_title="Smart Crop Advisory",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Smart Crop Advisory System")
st.write(
    "Crop Recommendation, Yield Prediction "
    "and Profitability Analysis"
)

st.success("Welcome! Your dashboard is running.")

st.subheader("Enter Soil Details")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)

with col2:
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)

with col3:
    potassium = st.number_input("Potassium (K)", min_value=0.0, value=43.0)

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0)

if st.button("Check Soil Details"):
    st.write("### Entered Values")
    st.json({
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "pH": ph,
        "Rainfall": rainfall
    })
