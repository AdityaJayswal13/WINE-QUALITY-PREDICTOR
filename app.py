import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

st.title('Wine Quality Predictor')

alcohol = st.number_input('Alcohol')

sulphate = st.number_input('Sulphate')

volatile = st.number_input('Volatile Acidity')

tsd = st.number_input('Sulphur dioxide')

density = st.number_input('Density')

if st.button('Predict Quality'):
    query = np.array([alcohol,sulphate,volatile,tsd,density])
    st.title("The Predicted Quality: "+str(model.predict([query])))