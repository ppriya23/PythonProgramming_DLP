import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scalr

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# selected features for prediction 

features ={
    "Age": {"type": "slider", "min": 5, "max": 80, "help": "Patient age in years"},
    "Gender":{"type": "select", "options":[0, 1], "help":"0 = Male , 1 = Female"},
    "Ethnicity": {"type": "select", "options": [0, 1, 2, 3], "help": "Categorical encoding of ethnicity"},
    "EducationLevel":{"type": "slider", "min":0, "max":3, "help":"0 = None ,1= Highschool, 2= Bachelors, 3= Higher"},
    "BMI": {"type": "slider", "min": 15, "max": 40, "help": "Body Mass Index"},
    "Smoking":{"type": "select", "options":[0, 1], "help":"0 = No smoking , 1 = Yes"},
    "PhysicalActivity": {"type": "slider", "min": 0, "max": 10, "help": "0 = None, 10 = Very active"},
    "DietQuality": {"type": "slider", "min": 0, "max": 10, "help": "0 = Poor, 10 = Excellent"},
    "SleepQuality": {"type": "slider", "min": 4, "max": 10, "help": "4 = Poor, 10 = Excellent"},
    "PollutionExposure": {"type": "slider", "min": 0, "max": 10, "help": "0 = None, 10 = High"},
    "PollenExposure":{"type": "slider", "min":0, "max":10, "help":"0 = None , 10 = high exposure"},
    "DustExposure":{"type": "slider", "min":0, "max":10, "help":"0 = None , 10 = high exposure"},
    "PetAllergy": {"type": "select", "options":[0, 1], "help":"0 = No pet allergy , 1 = Yes"},
    "FamilyHistoryAsthma":{"type": "select", "options":[0, 1], "help":"0 = No family history of asthma , 1 = Yes"},
    "HistoryOfAllergies": {"type": "select", "options":[0, 1], "help": "0 = No allergy history, 1 = Yes"},
    "Eczema": {"type": "select", "options":[0, 1], "help":"0 = No eczema , 1 = Yes"},
    "HayFever": {"type": "select", "options":[0, 1], "help":"0 = No hayfever , 1 = Yes"},
    "GastroesophagealReflux": {"type": "select", "options": [0, 1], "help": "0 = No, 1 = Yes"},
    "LungFunctionFEV1": {"type": "slider", "min": 1, "max": 4, "help": "Forced Expiratory Volume (L)"},
    "LungFunctionFVC": {"type": "slider", "min": 1, "max": 6, "help": "Forced Vital Capacity (L)"},
    "Wheezing": {"type": "select", "options":[0,1], "help": "0 =  NO wheezing, 1 = Yes"},
    "ShortnessOfBreath":{"type": "select", "options":[0, 1], "help":"0 = No shortness of breath , 1 = Yes"},
    "ChestTightness": {"type": "slider", "min":0, "max":10, "help":"0 = None , 10 = severe"},
    "Coughing":{"type": "select", "options":[0, 1], "help":"0 = No coughing , 1 = Yes"},
    "NighttimeSymptoms": {"type": "select", "options":[0, 1], "help":"0 = No night time symptoms , 1 = yes"},
    "ExerciseInduced": {"type": "select", "options": [0, 1], "help": "0 = No, 1 = Yes"}
   }


#App page

st.set_page_config(page_title="Asthma Risk Predictor", layout ="centered")
st.title("Asthma Risk Predictor")
st.markdown("Enter patient details below. Please look at the format before enter value.")

# Input form
with st.form("Prediction_form"):
    inputs ={}
    for name, config in features.items():
        if config["type"]=="select":
            inputs[name]= st.selectbox(name, config["options"], help = config["help"])
        elif config["type"]=="slider":
            inputs[name]= st.slider(name,config["min"], config["max"], help= config["help"])
    threshold = 0.3 #fixed in asthma.py code for recall prioritization in medical context
    submit = st.form_submit_button("Predict")

#Prediction

if submit:
    missing= [feature for feature, value in inputs.items() if value is None]
    if missing:
        st.error(f"Missing input values: {', '.join(missing)}. Please fill/refill them correctly.")
    else:
        try:
            input_df = pd.DataFrame([inputs])[list(features.keys())] 
            input_scaled = scaler.transform(input_df)
            prob= model.predict_proba(input_scaled)[0][1]
            pred = "Asthma" if prob >= threshold else "No Asthma"

            st.subheader("Asthma Prediction Result")
            st.metric("Asthma Risk Score", f"{prob:.2f}")
            st.write(f"Prediction (Threshold {threshold}): **{pred}**")
            st.markdown("This is not a medical report. This app is for educational purposes. Please consult a doctor for professional advice.")

            #Feature Contributions

            coef = model.coef_[0]
            contributions=pd.Series(input_scaled[0]* coef, index =input_df.columns).sort_values(key=abs,ascending=False)
            st.subheader("Feature Contributions")
            st.bar_chart(contributions)

            #Download result
            result_df = input_df.copy()
            result_df["RiskScore"]= prob
            result_df["Prediction"]= pred
            st.download_button("Download Result", result_df.to_csv(index=False), "asthma_prediction.csv", "text/csv")
        except Exception as e:
            st.error(f"Something went wrong : {e}. Please check your input value and try again.")


