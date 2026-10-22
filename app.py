import pandas as pd
import streamlit as st
import pickle


try:
    with open("final_model_.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please upload `final_model.pkl`.")
    model = None


st.markdown("<h1 class='title'>🏡 Health Score Predictor</h1>", unsafe_allow_html=True)


with st.expander("🔹 **Health Details**", expanded=True):
    Age	= st.number_input("Age",min_value=1.104792, max_value=86.232778, value=50.000000)
    BMI = st.number_input("BMI", min_value=10.298057, max_value=40.965538, value=28.000000	)
    Exercise_Frequency = st.number_input("Exercise_Frequency", min_value=1.000000	, max_value=6.000000, value=3.000000	)
    Diet_Quality = st.number_input("Diet_Quality", min_value=59.945481, max_value=110.265186, value=82.000000	)
    Sleep_Hours = st.number_input("Sleep_Hours", min_value=5.903351, max_value=11.638962, value=8.000000)
    Smoking_Status = st.number_input("Smoking_Status",min_value=0.000000, max_value=1.000000, value=0.500000)
    Alcohol_Consumption = st.number_input("Alcohol_Consumption", min_value=1.644111, max_value=11.105100, value=8.000000)
    
   

if st.button("🔍 Predict Health Score"):
    input_data = pd.DataFrame([[
        Age, BMI, Exercise_Frequency, Diet_Quality, Sleep_Hours,
       Smoking_Status, Alcohol_Consumption
                               ]],
                              columns=['Age', 'BMI', 'Exercise_Frequency', 'Diet_Quality', 'Sleep_Hours',
       'Smoking_Status', 'Alcohol_Consumption'])



    try:
        predicted_health_score = model.predict(input_data)[0]
        st.markdown(f"<div class='result-box'>🏠 Predicted Health Score:  {predicted_health_score:.2f} </div>", unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"❌ Error during prediction: {e}")