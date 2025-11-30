import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="ML Data Analyzer", layout="wide")

st.title("📊 Machine Learning Data Analyzer")
st.write("Upload a CSV → Explore Data → Train Model → Make Predictions")

# -------------------------
# Upload CSV
# -------------------------
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📁 Dataset Preview")
    st.dataframe(df)

    st.subheader("📌 Basic Information")
    st.write(df.describe())

    # Plot
    st.subheader("📈 Visualize Columns")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col = st.selectbox("Select numeric column to plot", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns found for plotting.")

    # -------------------------
    # ML Training
    # -------------------------
    st.subheader("🤖 Train Machine Learning Model")

    target = st.selectbox("Select target column", df.columns)

    if st.button("Train Model"):
        try:
            X = df.drop(columns=[target])
            X = pd.get_dummies(X, drop_first=True)
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"Model trained! 🎉 Accuracy: **{acc:.2f}**")

            st.session_state.model = model
            st.session_state.features = X.columns.tolist()

        except Exception as e:
            st.error(f"Error training model: {e}")

    # -------------------------
    # Prediction
    # -------------------------
    if "model" in st.session_state:
        st.subheader("🔮 Make Predictions")

        input_data = {}
        for col in st.session_state.features:
            input_data[col] = st.number_input(f"Value for {col}", 0.0)

        if st.button("Predict"):
            try:
                X_new = pd.DataFrame([input_data])
                prediction = st.session_state.model.predict(X_new)[0]
                st.success(f"Prediction: **{prediction}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
