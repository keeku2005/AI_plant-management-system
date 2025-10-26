import streamlit as st
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import csv
import pandas as pd
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
import requests  # To interact with the Bard API

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("dotenv module not found. Please make sure to install it using 'pip install python-dotenv'.")

# Define the path to the CSV file
csv_file_path = 'question_history.csv'

# Function to store questions in CSV file
def store_question(question):
    try:
        current_time = datetime.now()
        expiration_time = current_time + timedelta(days=7)  # Auto delete after a week

        # Format the data (question, expiration time)
        data = [question, expiration_time.strftime('%Y-%m-%d')]

        # Append the data to the CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as e:
        st.error(f"An error occurred while storing the question: {e}")

# Function to read question history from CSV file
def read_question_history():
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            history_data = list(reader)
        return history_data
    except FileNotFoundError:
        return []

# Function to call Bard API using only the API key
def ask_bard(query):
    API_KEY = "AIzaSyDnEhZ4EnphPTr5zOlvdWoZ9USC2HJHL9I" 

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query
    }

    response = requests.post(
        "https://bard.googleapis.com/v1/query",  # Fixed known endpoint for Bard
        json=payload, headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get('response', 'No response from Bard')
    else:
        return "Error: Unable to get response from Bard"

# Function to configure API key
def configure_api_key():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        st.warning(f"An error occurred while configuring the API key: {e}")

# Function for educational chat using Google Gemini Pro API
def gemini_pro(input_text, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt, input_text])
        return response.text
    except Exception as e:
        st.error(f"An error occurred during chat: {e}")

# Loading the Model and saving to cache using st.cache_resource
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)
    
    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Loading the Model
model = load_model('model_weights.h5')

# Title and Description
st.title('Plant Disease Detection and Chatbot')
st.write("Upload your plant's leaf image to get a health prediction, or ask our chatbot any questions about plant diseases!")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose an Image file", type=["png", "jpg"])

# If there is an uploaded file, start making predictions
if uploaded_file:
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    
    # Reading and displaying the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(image.resize((700, 400), Image.ANTIALIAS)), width=None)
    my_bar.progress(40)
    
    # Cleaning the image
    image = clean_image(image)
    
    # Making predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)
    
    # Making results
    result = make_results(predictions, predictions_arr)
    my_bar.progress(100)
    
    # Show the results
    st.write(f"The plant is {result['status']} with a {result['prediction']} prediction.")
    
    progress.empty()
    my_bar.empty()

# Chatbot Section
st.subheader('Chat with the Plant Disease Expert')

# Default prompt for generating responses
prompt = """Provide an informative and detailed response to the user's query."""

# Input field for user's question
input_text = st.text_input('Ask a question about plant diseases:')

# Button to send the question
if st.button('Send Message'):
    if input_text:
        configure_api_key()
        store_question(input_text)
        response = gemini_pro(input_text, prompt)
        st.write("Response:", response)
    else:
        st.warning("Please enter a question.")

# Display recent question history
history_data = read_question_history()
if history_data:
    st.subheader("Recent Search History")
    df = pd.DataFrame(history_data, columns=["Question", "Expire On"])
    st.write(df.tail(50).iloc[::-1])
else:
    st.info("No question history available.")
