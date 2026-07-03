import streamlit as st
import pickle
import json
import random

model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    data = json.load(file)

st.title("Customer Support Chatbot")

message = st.text_input("Ask your question")

if st.button("Send"):

    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]

    response = "Sorry, I don't understand."

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])

    st.write(response)