import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("ford.csv")

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("price", axis=1)
y = df["price"]

# =========================================================
# ONE HOT ENCODING
# =========================================================

X = pd.get_dummies(
    X,
    columns=['model', 'transmission', 'fuelType'],
    drop_first=True
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MODEL TRAINING
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred_test = model.predict(X_test)

r2 = r2_score(y_test, y_pred_test)

mae = mean_absolute_error(y_test, y_pred_test)

# =========================================================
# TITLE
# =========================================================

st.title("🚗 AI Powered Car Price Prediction System")

st.write(
    """
    Predict car prices using Machine Learning based on:
    model, mileage, fuel type, transmission, engine size, and more.
    """
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About Project")

st.sidebar.write(
    """
    This Machine Learning project predicts car prices using:

    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - Linear Regression
    """
)

st.sidebar.write(f"### Model Accuracy: {r2:.2f}")

st.sidebar.write(f"### MAE: {mae:,.2f}")

# =========================================================
# CAR IMAGE
# =========================================================

st.image(
    "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7",
    use_container_width=True
)

# =========================================================
# USER INPUTS
# =========================================================

st.subheader("Enter Car Details")

model_name = st.selectbox(
    "Select Model",
    sorted(df["model"].unique())
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
    value=2019
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

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Price"):

    # Create dataframe
    input_data = pd.DataFrame({
        'model': [model_name],
        'year': [year],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuel],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engineSize]
    })

    # One hot encode input
    input_data = pd.get_dummies(input_data)

    # Add missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[X.columns]

    # Prediction
    prediction = model.predict(input_data)

    # Currency conversion
    price_gbp = prediction[0]
    price_inr = price_gbp * 113

    # Output
    st.success(
        f"""
        ### Predicted Car Price

        🇬🇧 £{price_gbp:,.2f}

        🇮🇳 ₹{price_inr:,.2f}
        """
    )

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("📊 Actual Price vs Predicted Price")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(y_test, y_pred_test)

# Perfect prediction line
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

ax.set_xlabel("Actual Price")

ax.set_ylabel("Predicted Price")

ax.set_title("Actual vs Predicted Car Prices")

st.pyplot(fig)

# =========================================================
# PRICE DISTRIBUTION
# =========================================================

st.subheader("📈 Price Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 5))

sns.histplot(df["price"], bins=50, kde=True, ax=ax2)

ax2.set_title("Distribution of Car Prices")

st.pyplot(fig2)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader("🔥 Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(8, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax3
)

st.pyplot(fig3)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

