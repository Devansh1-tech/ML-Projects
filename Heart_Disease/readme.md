# ❤️ Heart Disease Prediction App

A Machine Learning-based web application that predicts the risk of heart disease using user health data. This app is built using **Streamlit** and provides real-time predictions along with probability scores.

---

## 🚀 Features

- Predicts **Heart Disease Risk (High / Low)**
- Displays **Risk Probability (%)**
- Clean and interactive UI using Streamlit
- Handles **data preprocessing (scaling + encoding)** internally
- Uses a trained **K-Nearest Neighbors (KNN)** model
- Fast and lightweight application

---

## 🧠 Machine Learning Pipeline

- **Dataset:** Heart Disease Dataset (`heart.csv`)
- **Preprocessing:**
  - One-Hot Encoding for categorical variables
  - Feature Scaling using StandardScaler
- **Model Used:**
  - K-Nearest Neighbors (KNN)
- **Saved Artifacts:**
  - `KNN_heart.pkl` → trained model
  - `scaler.pkl` → feature scaler
  - `columns.pkl` → column alignment

---

## 🖥️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib

---

## 📂 Project Structure
heart-disease-app/
│
├── app.py # Streamlit application
├── heart.csv # Dataset
├── Heart.ipynb # Model training notebook
├── KNN_heart.pkl # Trained ML model
├── scaler.pkl # Scaler object
├── columns.pkl # Feature columns
├── requirements.txt # Dependencies
└── README.md # Documentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/your-username/heart-disease-prediction.git

cd heart-disease-prediction

### 2️⃣ Install Dependencies


pip install -r requirements.txt


### 3️⃣ Run the Application


streamlit run app.py


---

## 📊 Input Features

The model takes the following inputs:

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate  
- Exercise-Induced Angina  
- Oldpeak (ST Depression)  
- ST Slope  
