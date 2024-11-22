#!/usr/bin/env python
# coding: utf-8

# In[59]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[61]:


with open("titanic_model.pkl", "rb") as file:
    model=pickle.load(file)


# In[63]:


model


# In[65]:


# Streamlit app
st.title("Titanic Survival Prediction")
st.write("Enter the passenger details to predict survival.")


# In[67]:


# User input
Pclass = st.selectbox("Pclass", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
Fare = st.slider("Fare", 0.0, 500.0, 30.0)
Embarked = st.selectbox("Embarked", ["C", "Q", "S"])


# In[69]:


# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "Fare": [Fare],
        "Embarked": [Embarked]
    })
    
    prediction = model.predict(input_data)
    survival_probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
    st.write(f"Survival Probability: {survival_probability:.2f}")


# # --- The End ----
