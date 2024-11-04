# Import libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load("best_model.pkl")  # Load the best model here
scaler = StandardScaler()

# Title and description
st.title("Real Estate Price Prediction App")
st.write("Predict the price of a house based on various features like bedroom count, space, etc.")

# Input fields for the features
def user_input_features():
    bedroom = st.number_input("Bedroom", min_value=0, max_value=10, value=3)
    space = st.number_input("Space (in square feet)", min_value=500, max_value=10000, value=1500)
    room = st.number_input("Room", min_value=0, max_value=10, value=5)
    lot = st.number_input("Lot Size", min_value=500, max_value=10000, value=2000)
    tax = st.number_input("Tax", min_value=0, max_value=10000, value=1500)
    bathroom = st.number_input("Bathroom", min_value=0, max_value=10, value=2)
    garage = st.number_input("Garage", min_value=0, max_value=5, value=1)

    data = {
        'Bedroom': bedroom,
        'Space': space,
        'Room': room,
        'Lot': lot,
        'Tax': tax,
        'Bathroom': bathroom,
        'Garage': garage
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Scaling data without polynomial transformation for GradientBoostingRegressor or similar models
input_df_scaled = scaler.fit_transform(input_df)

# Predict using the best model (assuming Gradient Boosting Regressor here)
prediction = model.predict(input_df_scaled)

# Display the result
st.subheader("Predicted Price")
st.write(f"${round(prediction[0], 2)}")

# R-squared score display
r2_score_best_model = 0.85  # Replace with the actual R2 score of the best model
st.subheader("Model Performance")
st.write(f"R-squared Score of the Best Model: {r2_score_best_model}")


