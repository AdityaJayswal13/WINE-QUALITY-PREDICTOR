import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('5_best.pkl', 'rb'))

st.sidebar.title('Wine Quality Predictor')

# Add some space between the title and the image in the sidebar
st.sidebar.write('')

image = "wines.jpg"
st.sidebar.image(image, use_column_width=True)

alcohol = st.sidebar.selectbox('Alcohol', df['alcohol'].unique())

sulphate = st.sidebar.selectbox('Sulphate', df['sulphates'].unique())

volatile = st.sidebar.selectbox('Volatile Acidity', df['volatile acidity'].unique())

tsd = st.sidebar.selectbox('Sulphur dioxide', df['total sulfur dioxide'].unique())

density = st.sidebar.selectbox('Density', df['density'].unique())

logo_image = "wine.png"
st.image(logo_image, use_column_width=True)

# About Our Model
st.markdown("---")
st.markdown("## About Our Model")
st.markdown("Our model is based on a machine learning algorithm (Random Forest) trained on a dataset containing various wine characteristics and corresponding quality ratings. It takes into account factors such as alcohol content, sulphate level, volatile acidity, total sulfur dioxide, density, and other indicators to predict the quality of wine. The model has been trained and fine-tuned to provide accurate predictions based on the input provided by the user.")
st.markdown("---")

if st.sidebar.button('Predict Wine Quality'):
    query = np.array([alcohol, sulphate, volatile, tsd, density])
    prediction = model.predict([query])[0]

if 'prediction' in locals():
    st.markdown("---")

    quality_categories = ["Worst", "Poor", "Below Average", "Average", "Above Average", "Good", "Excellent", "Best"]
    quality_index = min(max(int(prediction * 5), 0), 7)  # Scale prediction to range from 0 to 7
    quality = quality_categories[quality_index]
    st.info(f"### Predicted Wine Quality: {quality}")
