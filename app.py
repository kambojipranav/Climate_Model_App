# climate_modeling_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import base64
import smtplib
from email.message import EmailMessage
import joblib
import os

# -------------------- CONFIGURATION --------------------
st.set_page_config(layout="wide", page_title="Climate Change Emission Predictor")

# -------------------- HELPER FUNCTIONS --------------------
@st.cache_data

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

def preprocess(df):
    df = df.copy()
    df = df.dropna(axis=1, thresh=int(0.9*len(df)))
    df = df.select_dtypes(include=[np.number])
    return df

def build_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_emission(model, X):
    return model.predict(X)

def visualize_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')


def generate_map(df):
    fmap = folium.Map(location=[-0.5, 29.3], zoom_start=7)
    heat_data = [[row["latitude"], row["longitude"], row["emission"]] for index, row in df.iterrows() if not np.isnan(row["emission"])]
    HeatMap(heat_data).add_to(fmap)
    return fmap

def send_email(receiver_email, subject, body):
    EMAIL = os.environ.get("EMAIL")
    PASSWORD = os.environ.get("EMAIL_PASSWORD")
    if not EMAIL or not PASSWORD:
        st.error("Email credentials not set in environment variables.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = receiver_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)
        st.success("Email sent successfully!")

# -------------------- MAIN STREAMLIT APP --------------------

train_raw, test_raw = load_data()

st.title("🌍 Climate Change Modeling App")
st.markdown("Predict CO₂ emissions using ML + visualize hotspots + send results via email")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["EDA & Map", "Model Training", "Predict Emission", "Send Email"])

if section == "EDA & Map":
    st.header("📊 Exploratory Data Analysis + Map")
    st.write("Train shape:", train_raw.shape)
    st.write("Null values:", train_raw.isnull().sum().sort_values(ascending=False).head())
    st.subheader("📍 Geo Heatmap of Emissions")
    try:
        fmap = generate_map(train_raw[['latitude', 'longitude', 'emission']].dropna())
        folium_static(fmap)
    except Exception as e:
        st.error(f"Map failed: {e}")

elif section == "Model Training":
    st.header("🧠 Train ML Model")
    df = preprocess(train_raw)
    X = df.drop("emission", axis=1)
    y = df["emission"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = build_model(X_train, y_train)
    joblib.dump(model, "model/rf_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    y_pred = model.predict(X_test)
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.2f}")
    st.subheader("📈 SHAP Feature Importance")
    visualize_shap(model, pd.DataFrame(X_test, columns=X.columns))

elif section == "Predict Emission":
    st.header("🚀 Predict Emission")
    test_clean = preprocess(test_raw)
    model = joblib.load("model/rf_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    X_test = scaler.transform(test_clean)
    preds = model.predict(X_test)
    test_raw["predicted_emission"] = preds
    st.dataframe(test_raw[["latitude", "longitude", "predicted_emission"]].head(20))
    st.download_button("Download Predictions", data=test_raw.to_csv(index=False), file_name="predictions.csv")

elif section == "Send Email":
    st.header("📧 Send Email with Prediction")
    email_to = st.text_input("Receiver Email")
    email_subject = st.text_input("Subject", value="Climate Emission Prediction Results")
    email_body = st.text_area("Email Body", value="Please find the attached prediction summary.")
    if st.button("Send Email"):
        send_email(email_to, email_subject, email_body)
