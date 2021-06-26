import pandas as pd
import joblib
import re
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

lemmatizer = WordNetLemmatizer()

def get_text():
  input_text = ["what are you", "hi"]
  df_input = pd.DataFrame(input_text, columns=['questions'])
  print(df_input)
  return df_input

from tensorflow.keras.models import load_model

#load the model
model = load_model('model-v1.h5')
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.plk')

def tokenizer(entry):
  tokens = entry.split()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  tokens = [re_punc.sub('', w) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
#     stop_words = set(stopwords.words('english'))
#     tokens = [w for w in tokens if not w in stop_words]
  tokens = [word.lower() for word in tokens if len(word) > 1]
  return tokens

def remove_stop_words_for_input(tokenizer,df,feature):
  doc_without_stopwords = []
  entry = df[feature][0]
  tokens = tokenizer(entry)
  doc_without_stopwords.append(' '.join(tokens))
  df[feature] = doc_without_stopwords
  return df

def encode_input_text(tokenizer_t,df,feature):
  t = tokenizer_t
  entry = entry = [df[feature][0]]
  encoded = t.texts_to_sequences(entry)
  padded = pad_sequences(encoded, maxlen=16, padding='post')
  return padded


def get_pred(model,encoded_input):
  pred = np.argmax(model.predict(encoded_input))
  return pred


def bot_precausion(df_input,pred):
  words = df_input.questions[0].split()
  if len([w for w in words if w in vocab])==0 :
      pred = 1
  return pred


def get_response(df2,pred):
  upper_bound = df2.groupby('labels').get_group(pred).shape[0]
  r = np.random.randint(0,upper_bound)
  responses = list(df2.groupby('labels').get_group(pred).response)
  return responses[r]

def bot_response(response,):
  print(response)


df_input = get_text()

#load artifacts 
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.plk')

df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

pred = get_pred(model,encoded_input)
pred = bot_precausion(df_input,pred)

response = get_response(df2,pred)
bot_response(response)