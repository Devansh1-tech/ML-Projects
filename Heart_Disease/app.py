import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "KNN_heart.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "columns.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, expected_columns


def build_input_frame(
    expected_columns,
    age,
    sex,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
):
    row = {column: 0 for column in expected_columns}
    row["Age"] = age
    row["RestingBP"] = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"] = fasting_bs
    row["MaxHR"] = max_hr
    row["Oldpeak"] = oldpeak

    encoded_values = {
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    for column, value in encoded_values.items():
        if column in row:
            row[column] = value

    return pd.DataFrame([row], columns=expected_columns)


model, scaler, expected_columns = load_artifacts()

st.title("Heart Disease Prediction")
st.markdown("Provide the details below to estimate heart disease risk.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.5, 1.0, 0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_df = build_input_frame(
        expected_columns,
        age,
        sex,
        chest_pain,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_slope,
    )
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")
