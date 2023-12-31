import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define a function to get predictions
def get_prediction(text):
    result = classifier(text)[0]
    return result['label'], result['score']

# Create a Streamlit app
st.title("Text Classification")
st.write("Input text to see classification predictions based on a pre-trained text classification model from Hugging Face.")

# Get the user input for the text
text = st.text_area("Enter your text here:")

# A button for the prediction
if st.button("Enter"):
    # Check if the text is not empty
    if text:
        # Get the prediction
        label, score = get_prediction(text)
        # Display the prediction
        st.write(f"The predicted label is: {label} with a score of {score}")
