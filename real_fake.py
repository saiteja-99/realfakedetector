import streamlit as st
import numpy as np
import joblib
vect=joblib.load("vectorizer")        #loading the vectorizer 
st.title("REAL-FAKE Classifier")      #title of the streamlit page
test_model=joblib.load("REAL-FAKE")   #loading the model (deserialization)
ip=st.text_input("Enter the message") #for input text to classify
op=np.array([ip])                     #making input text as numpy array
op=vect.transform(op)                 #transforming the input text using vectorizer
res=test_model.predict(op)            #op is the transformed text to be predicted
if st.button('predict'):               #if button is active
  #st.write(ip)
  st.title(res[0])                    #for displaying REAL (or) FAKE 