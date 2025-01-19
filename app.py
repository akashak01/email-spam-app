import streamlit as st 
import pickle

model=pickle.load(open('spam1.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("This is a machine LEarning project used to Classify the email as spam and not spam")

user_input=st.text_area("Enter email to classify", height=150)

if st.button("Classify"):
    if user_input:
        data=[user_input]
        vectorized_data=cv.transform(data).toarray()
        result=model.predict(vectorized_data)
        if result[0]==0:
            st.write("The email is not spam")
        else :
            st.write("The Email is spam")
    else :
        st.write("Please type email to classify")            