import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Setting page layout
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Loading dataset
df = pd.read_csv("ford.csv")

# Input features and target column
X = df.drop("price", axis=1)
y = df["price"]

# Encoding categorical columns
X = pd.get_dummies(
    X,
    columns=["model", "transmission", "fuelType"],
    drop_first=True
)

# Splitting dataset into training and testing data
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

# Predictions on test data
y_pred_test = model.predict(X_test)

# Model evaluation metrics
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

# App title
st.title(" AI Powered Car Price Prediction System")

st.write(
    """
Predict car prices using Machine Learning based on:
model, mileage, fuel type, engine size and more.
"""
)

# Sidebar section
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

st.sidebar.success("Model Used: Random Forest Regressor")

st.sidebar.write(f"### R² Score: {r2:.2f}")

st.sidebar.write(f"### MAE: {mae:,.2f}")

st.sidebar.write(f"### RMSE: {rmse:,.2f}")

# Workflow explanation
st.sidebar.subheader("⚙️ Project Workflow")

st.sidebar.write(
    """
1. Data Collection  
2. Data Preprocessing  
3. One Hot Encoding  
4. Model Training  
5. Prediction  
6. Visualization
"""
)

# Local car images
car_images = {

    "Mustang": "images/Mustang.jpg",

    "Focus": "images/Focus.jpg",

    "Fiesta": "images/Fiesta.jpg",

    "Kuga": "images/Kuga.jpg",

    "EcoSport": "images/ecosport.jpg"
}

# User input section
st.subheader("Enter Car Details")

model_name = st.selectbox(
    "Select Model",
    sorted(df["model"].unique())
)

# Displaying selected car image
if model_name in car_images:

    st.image(
        car_images[model_name],
        width=700
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

    # Creating dataframe from user inputs
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

    # Matching training column order
    input_data = input_data[X.columns]

    # Predicting car price
    prediction = model.predict(input_data)

    price_gbp = prediction[0]

    price_inr = price_gbp * 113

    # Displaying prediction
    st.success(
        f"""
## Predicted Car Price

🇬🇧 £{price_gbp:,.2f}

🇮🇳 ₹{price_inr:,.2f}
"""
    )

    # Comparing prediction with average market price
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

    fig_compare = px.bar(
        comparison_df,
        x="Type",
        y="Price",
        color="Type",
        template="plotly_dark",
        text_auto=True
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

# Actual vs predicted graph
st.subheader("📊 Actual Price vs Predicted Price")

actual_vs_pred = pd.DataFrame({

    "Actual Price": y_test,

    "Predicted Price": y_pred_test
})

fig1 = px.scatter(
    actual_vs_pred,
    x="Actual Price",
    y="Predicted Price",
    template="plotly_dark",
    opacity=0.7
)

fig1.add_trace(
    go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode="lines",
        name="Perfect Prediction"
    )
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# Price distribution graph
st.subheader("📈 Price Distribution")

fig2 = px.histogram(
    df,
    x="price",
    nbins=50,
    template="plotly_dark"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig3 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    template="plotly_dark"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

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

fig4 = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    template="plotly_dark",
    color="Importance"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# Sample dataset preview
st.subheader("🚘 Sample Predictions")

sample_df = df[[
    "model",
    "year",
    "mileage",
    "fuelType",
    "price"
]].sample(5)

st.dataframe(
    sample_df,
    use_container_width=True
)

