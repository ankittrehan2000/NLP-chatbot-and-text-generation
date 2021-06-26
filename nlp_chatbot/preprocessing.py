import pandas as pd
import re
import string
import os
import json
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
import joblib
from tensorflow.keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPool1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def load_doc(jsonFile):
  with open(jsonFile) as file:
    json_data = json.loads(file.read())
  return json_data
dir_path = os.path.dirname(os.path.realpath(__file__))
data = os.path.join(dir_path, "intents.json")
data = load_doc(data)

#frame json file with pandas
def frame_data(feat_1, feat_2, is_pattern):
  is_pattern = is_pattern
  df = pd.DataFrame(columns=[feat_1, feat_2])
  #for every type of question to expect in data (the json file)
  for intent in data['intents']:
    if is_pattern:
      for pattern in intent['patterns']:
        word = pattern
        df_to_append = pd.Series([word, intent['tag']], index=df.columns)
        df = df.append(df_to_append, ignore_index=True)
    else:
      for response in intent['responses']:
        word = response
        df_to_append = pd.Series([word, intent['tag']], index=df.columns)
        df = df.append(df_to_append, ignore_index=True)
  return df

df1 = frame_data('questions', 'labels', True)
df2 = frame_data('responses', 'labels', False)

#if error for not having stopwords downloaded in nltk
#nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
vocab = Counter()
labels = []

#split into words and give words numbers
def tokenizer(entry):
  tokens = entry.split()
  #remove all punctuation using regex
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  tokens = [re_punc.sub('', w) for w in tokens]
  #.isAlpha returns true if string is an alphabetic string
  tokens = [word for word in tokens if word.isalpha()]
  #lemmatization - replace words with their root form
  tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
  tokens = [word.lower() for word in tokens]
  return tokens

#remove stop words - words that do not contribute to meaning of phrase
#examples - pronouns, indefinite articles etc
def remove_stop_words(tokenizer, df, feature):
  doc_without_words = []
  for entry in df[feature]:
    tokens = tokenizer(entry)
    joblib.dump(tokens, 'tokens.plk')
    doc_without_words.append(' '.join(tokens))
  df[feature] = doc_without_words
  return None

#build a vocabulary - set of words in a dataset
def create_vocab(tokenizer, df, feature):
  for entry in df[feature]:
    tokens = tokenizer(entry)
    vocab.update(tokens)
  joblib.dump(tokens, 'vocab.plk')
  return None

#call the above functions to create a datadump
create_vocab(tokenizer, df1, 'questions')
remove_stop_words(tokenizer, df1, 'questions')

test_list = list(df1.groupby(by='labels', as_index=False).first()['questions'])
test_index = []
for i,_ in enumerate(test_list):
  idx = df1[df1.questions == test_list[i]].index[0]
  test_index.append(idx)
train_index = [i for i in df1.index if i not in test_index]

#convert to a sequence of integers
def encoder(df, feature):
  t = Tokenizer()
  entries = [entry for entry in df[feature]]
  #updates internal vocab based on texts
  t.fit_on_texts(entries)
  joblib.dump(t, 'tokenizer_t.pkl')
  vocab_size = len(t.word_index) + 1
  max_length = max([len(s.split()) for s in entries])
  #convert the text in the entires to numerical sequences for the model
  encoded = t.texts_to_sequences(entries)
  padded = pad_sequences(encoded, maxlen=max_length, padding='post')
  return padded, vocab_size

#encode and use that encoded value returned for creating another dataframe
X, vocab_size = encoder(df1, 'questions')
df_encoded = pd.DataFrame(X)
df_encoded['labels'] = df1.labels

for i in range(0, 2):
  dt = [0]*16
  dt.append('confused')
  dt = [dt]
  pd.DataFrame(dt).rename(columns = {16:'labels'})
  df_encoded = df_encoded.append(pd.DataFrame(dt).rename(columns = {16:'labels'}),ignore_index=True)

#show end of tail
df_encoded.tail()

#add two rows from the preivious code in the df and use those row numbers to add to the index
train_index.append(87)
test_index.append(88)

label_enc = LabelEncoder()
#used to normalize labels
labl = label_enc.fit_transform(df_encoded.labels)

mapper = {}
for index, key in enumerate(df_encoded.labels):
  if key not in mapper.keys():
    mapper[key] = labl[index]

df2.labels = df2.labels.map(mapper).astype({'labels': 'int32'})
df2.to_csv('response.csv', index = False)

train = df_encoded.loc[train_index]
test = df_encoded.loc[test_index]

#set the data to be in x and y format
x_train = train.drop(columns=['labels'], axis=1)
y_train = train.labels
x_test = test.drop(columns=['labels'], axis=1)
y_test = test.labels

#convert to indicator variables
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

max_length = x_train.shape[1]

#to halt the training of the network at the perfect time
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint("model-v1.h5",
  monitor='val_loss',
  mode='min',
  save_best_only=True,
  verbose=1
)

#learning rate is reduced if there are no improvements seen in the past iterations
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
callbacks = [early_stopping, checkpoint, reducelr]

def define_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 300, input_length=max_length))
  model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
  model.add(MaxPool1D(pool_size=8))
  model.add(Flatten())
  model.add(Dense(17, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

model = define_model(vocab_size, max_length)
history = model.fit(x_train, y_train, epochs = 500, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)