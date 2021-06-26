import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#characters in the text that was imported
vocab = sorted(set(text))

#set up the conversion to numerical representation
ids_from_chars = preprocessing.StringLookup(
  vocabulary = list(vocab), mask_token=None
)
#set up the reverse for human readable format
chars_from_ids = preprocessing.StringLookup(
  vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#goal - given a character or a sequence of characters, what is the most probable next character
#convert the text to ids
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
#convert to a stream of text - logical entity that can accept input or output
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)//(seq_length + 1)
#convert the individual characters to sequences of desired size
sequences = ids_dataset.batch(seq_length + 1, drop_remainder = True)

#create the training data
def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text

#example for what above does:
#split_input_target(list("Tensorflow"))
#(['T', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o'],
# ['e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w'])

dataset = sequences.map(split_input_target)
#example of dataset element
#for input_example, target_example in dataset.take(1):
#    print("Input :", text_from_ids(input_example).numpy())
#    print("Target:", text_from_ids(target_example).numpy())

#shuffle the data
BATCH_SIZE = 64
#buffer size is needed to shuffle the dataset because it maintains a buffer of elements and shuffles elements in it
BUFFER_SIZE = 10000
dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True).prefetch(tf.data.experimental.AUTOTUNE))
#above creates a dataset of shape (64, 100) (64, 100)

#define the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

#rnn - recurrent neural networks - specify hidden states that do not depend on the input 
#gru is a type or rnn 
class FirstModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    #converts indexes into dense vectors - a vector where most of the values are non-zero
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

#train and set up the model
model = FirstModel(
  #vocabulary size should match the size of the training set 
  vocab_size = len(ids_from_chars.get_vocabulary()),
  embedding_dim = embedding_dim,
  rnn_units = rnn_units
)

#prepare to get actual predictions from the model
#compile the model
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

#generate predictions
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    #create a mask to prevent unknown outputs
    skip_ids = self.ids_from_chars(['UNK'])[:, None]
    sparse_mask = tf.SparseTensor(
      values = [-float('inf')] * len(skip_ids),
      indices = skip_ids,
      dense_shape = [len(ids_from_chars.get_vocabulary())]
    )
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
      #convert the strings
      input_chars = tf.strings.unicode_split(input, 'UTF-8')
      input_ids = self.ids_from_chars(input_chars).to_tensor()

      #run the model and 
      predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)

      #only use the last prediction
      predicted_logits = predicted_logits[:, -1, :]
      predicted_logits = predicted_logits/self.temperature
      # Apply the prediction mask: prevent "[UNK]" from being generated.
      predicted_logits = predicted_logits + self.prediction_mask
      #sample the outputs to get the id
      predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
      predicted_ids = tf.squeeze(predicted_ids, axis=-1)

      #convert from token ids to characters
      predicted_chars = self.chars_from_ids(predicted_ids)
      return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
#run in a for loop to generate some text
start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]
for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)