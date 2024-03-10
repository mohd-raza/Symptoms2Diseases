import streamlit as st
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the trained model
model_embed = load_model('./embed.h5')

# Load the data to get the labels
path = "Symptom2Disease.csv"
data = pd.read_csv(path)

# Fit the label encoder
label_encoder = LabelEncoder().fit(data['label'])

# Define the tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])

# Define the function to remove stopwords
def remove_stopwords(texts):
  words = [re.split(r'[ ,.]+', text) for text in texts]
  words = [[word.lower() for word in text if word != ''] for text in words]
  stop_words = set(stopwords.words('english'))
  words = [[word for word in text if word.lower() not in stop_words] for text in words]
  return words

# Define the function to tokenize and pad
def tokenize_and_pad(words,tokenizer, max_len=50):
    sequences = tokenizer.texts_to_sequences(words)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Streamlit code
st.title('Symptom to Disease Predictor')

st.sidebar.title('About')
st.sidebar.info('This application is built with Streamlit and uses a LSTM model to predict possible diseases based on the symptoms entered by the user. The model was trained on a dataset of symptoms and corresponding diseases. The top 3 possible diseases are displayed as output.')

user_input = st.text_input("Enter your symptoms:")

if st.button('Predict'):
    test_text = [user_input]
    test_sequences = tokenize_and_pad(remove_stopwords(test_text), tokenizer)
    test_pred = model_embed.predict(test_sequences)
    print(test_pred)
    top= np.argsort(test_pred[0])[-1:][::-1]
    x = [test_pred[0][i] for i in top]
    y = (label_encoder.inverse_transform(top))
    st.write('Based on your symptoms, the most probable disease is:', y[0])
