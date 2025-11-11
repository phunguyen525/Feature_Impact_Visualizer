# feature_impact_visualizer.py
import shap
import streamlit as st
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = load_boston()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

st.title("🏠 Feature Impact Visualizer")
st.write("This dashboard explains how each feature affects house price predictions using SHAP values.")

st.subheader("Feature Importance Overview")
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
st.pyplot(bbox_inches="tight")

st.subheader("Single Prediction Explanation")
index = st.slider("Select a sample:", 0, len(X_test)-1, 0)
shap.force_plot(explainer.expected_value, shap_values[index, :], feature_names=feature_names, matplotlib=True, show=False)
st.pyplot(bbox_inches="tight")
