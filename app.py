import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Page setup
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Reading dataset
df = pd.read_csv("ford.csv")

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Converting categorical data into numbers
X = pd.get_dummies(
    X,
    columns=["model", "transmission", "fuelType"],
    drop_first=True
)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction on test data
y_pred_test = model.predict(X_test)

# Model performance
r2 = r2_score(y_test, y_pred_test)

mae = mean_absolute_error(
    y_test,
    y_pred_test
)

mse = mean_squared_error(
    y_test,
    y_pred_test
)

rmse = mse ** 0.5

# Main heading
st.title(" AI Powered Car Price Prediction System")

st.write(
    "Predict car prices using Machine Learning based on mileage, fuel type, engine size and more."
)

# Sidebar info
st.sidebar.title(" About Project")

st.sidebar.write(
    """
This project predicts car prices using:

• Python  
• Pandas  
• Scikit-Learn  
• Streamlit  
• Random Forest Regressor
"""
)

st.sidebar.write(f"### R² Score: {r2:.2f}")

st.sidebar.write(f"### MAE: {mae:,.2f}")

st.sidebar.write(f"### RMSE: {rmse:,.2f}")

st.sidebar.success("Model Used: Random Forest Regressor")

# Local images
car_images = {

    "Mustang": "images/Mustang.webp",

    "Focus": "images/focus.webp",

    "Fiesta": "images/fiesta.webp",

    "Kuga": "images/kuga.webp"
}

# User inputs
st.subheader("Enter Car Details")

model_name = st.selectbox(
    "Select Model",
    sorted(df["model"].unique())
)

# Showing image according to selected model
if model_name in car_images:

    st.image(
        car_images[model_name],
        use_container_width=True
    )

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["fuelType"].unique())
)

year = st.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2025,
    value=2021
)

mileage = st.number_input(
    "Enter Mileage",
    min_value=0,
    value=15000
)

tax = st.number_input(
    "Enter Tax",
    min_value=0,
    value=145
)

mpg = st.number_input(
    "Enter MPG",
    min_value=0.0,
    value=55.0
)

engineSize = st.number_input(
    "Enter Engine Size",
    min_value=0.0,
    value=1.5
)

# Predict button
if st.button("Predict Price"):

    # Creating dataframe from user input
    input_data = pd.DataFrame({

        "model": [model_name],

        "year": [year],

        "transmission": [transmission],

        "mileage": [mileage],

        "fuelType": [fuel],

        "tax": [tax],

        "mpg": [mpg],

        "engineSize": [engineSize]
    })

    # Encoding user input
    input_data = pd.get_dummies(input_data)

    # Adding missing columns
    for col in X.columns:

        if col not in input_data.columns:

            input_data[col] = 0

    # Matching column order
    input_data = input_data[X.columns]

    # Final prediction
    prediction = model.predict(input_data)

    price_gbp = prediction[0]

    price_inr = price_gbp * 113

    # Displaying output
    st.success(
        f"""
## Predicted Car Price

🇬🇧 £{price_gbp:,.2f}

🇮🇳 ₹{price_inr:,.2f}
"""
    )

    # Comparison graph
    comparison_df = pd.DataFrame({

        "Type": [
            "Predicted Price",
            "Average Market Price"
        ],

        "Price": [
            price_gbp,
            df["price"].mean()
        ]
    })

    st.subheader("📊 Predicted Price Comparison")

    fig5, ax5 = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x="Type",
        y="Price",
        data=comparison_df,
        ax=ax5
    )

    ax5.set_title("Predicted vs Average Price")

    st.pyplot(fig5)

# Actual vs predicted graph
st.subheader("📊 Actual Price vs Predicted Price")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    y_test,
    y_pred_test
)

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

ax.set_xlabel("Actual Price")

ax.set_ylabel("Predicted Price")

ax.set_title("Actual vs Predicted Car Prices")

st.pyplot(fig)

# Distribution graph
st.subheader("📈 Price Distribution")

fig2, ax2 = plt.subplots(figsize=(10, 5))

sns.histplot(
    df["price"],
    bins=50,
    kde=True,
    ax=ax2
)

ax2.set_title("Distribution of Car Prices")

st.pyplot(fig2)

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax3
)

st.pyplot(fig3)

# Feature importance graph
st.subheader("⭐ Feature Importance")

importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig4, ax4 = plt.subplots(figsize=(10, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance,
    ax=ax4
)

ax4.set_title("Top 10 Important Features")

st.pyplot(fig4)