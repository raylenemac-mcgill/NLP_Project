import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, Input, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence

f = open("rt-polaritydata/rt-polarity.neg", "rb")
phrases = f.read().splitlines()
f = open("rt-polaritydata/rt-polarity.pos", "rb")
phrases.extend(f.read().splitlines())
print(phrases[:5])

labels = [0 for _ in range(5331)] + [1 for _ in range(5331)]
x_train, x_test, y_train, y_test = train_test_split(phrases, labels, test_size=0.33)
x_train = [str(sent) for sent in x_train]
x_test = [str(sent) for sent in x_test]
print(x_train[:5])

tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
tokenizer.fit_on_texts(x_train)
vocab_size = len(tokenizer.word_index) + 1
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print(len(x_train))
print(x_train[0], x_train[1])
print(len(x_test))
print(x_test[0], x_test[1])

x_train = sequence.pad_sequences(x_train, maxlen=40, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=40, padding='post', truncating='post')
print(x_train[:3])
print(len(x_train[0]), len(x_train[1]), len(x_train[2]))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

dims = 100
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded this many word vectors: ', len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, dims))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

import keras
print(keras.backend.image_data_format())

mod2 = Sequential()
mod2.add(Embedding(input_dim=vocab_size,output_dim=dims,weights=[embedding_matrix],input_length=len(x_train[0]),trainable=True))
mod2.add(Flatten())
mod2.add(Dense(128, activation='relu'))
mod2.add(Dropout(0.2))
mod2.add(Dense(32, activation='relu'))
mod2.add(Dropout(0.2))
mod2.add(Dense(1, activation='sigmoid'))
# compile the model
mod2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(mod2.summary())

mod2.fit(x_train, y_train, epochs=5)
# evaluate the model
loss, accuracy = mod2.evaluate(x_test, y_test)
print('\nAccuracy: %f' % (accuracy*100))

