import streamlit as st
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from litellm import completion
import os
os.environ["OPENAI_API_KEY"]= ""

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
img = "./dataset-cover.jpg"
st.image(img, use_column_width=True)
st.title('DiagnoSense: Symptoms Simplified, Solutions Amplified!')

st.sidebar.title('About DiagnoSense')
st.sidebar.info('This application is built with Streamlit and uses a Custom Trained Natural Language Processing model to predict possible diseases based on the symptoms entered by the user. The model was trained on a dataset of symptoms and corresponding diseases.')

def predict_disease(user_input):
    test_text = [user_input]
    test_sequences = tokenize_and_pad(remove_stopwords(test_text), tokenizer)
    test_pred = model_embed.predict(test_sequences)
    
    top= np.argsort(test_pred[0])[-3:][::-1]
    x = [test_pred[0][i] for i in top]
    print(x)
    y = (label_encoder.inverse_transform(top))
    print(y)
    return y[0].capitalize()

# Chatbot function using OpenAI LLM
def chat_with_llm(user_input):
    system_prompt= """
    You are a helpful Medical Assistant who has a experience of more than 10+ years in the medical field and knowledge of more than 500 diseases and their symptoms. 
    
    Your main task is to help the user answer any question based on the Symptom.
    
    You should strictly not provide or not talk about any other diseases based on the symptoms provided by the user.

    Your response should be helpful and informative and provide user with some immediate solutions to the symptoms which can be performed without going to a doctor or a hospital. 
        
    You should ask the user for more information if only needed otherwise no need to ask any questions.
    
    You should have human tone in your response.
    """
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    response = completion(
      model="gpt-3.5-turbo",
      messages=messages,
      temperature=0.6
    )
    return response.choices[0].message['content']
user_input = st.text_input("Enter your symptoms:")

set_in_pred = False
if st.button('Predict'):
    set_in_pred = True
    disease_prediction = predict_disease(user_input)
    st.write('Based on your symptoms, the most probable disease is:', disease_prediction)


if st.sidebar.selectbox('More Information', ['No', 'Yes']) == 'Yes':

    st.subheader("DiagnoSense Bot")
    chat_input = st.text_area("You:", "")
    if st.button("Send"):
        bot_response = chat_with_llm(chat_input)
        st.text_area("DiagnoSense Bot:", value=bot_response, height=200, max_chars=None, key=None)