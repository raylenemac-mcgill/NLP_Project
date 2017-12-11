import cPickle
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, Input, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split

with open(r"vectors.pickle", "rb") as input_file:
	vecs = cPickle.load(input_file)

half_split = len(vecs)/2
labels = np.zeros(half_split + half_split)
labels[:half_split] = 1.0
labels[half_split:] = 0.0
print(half_split)

seed = 7
x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.3, random_state=seed)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed)

mod2 = Sequential()
mod2.add(Dense(128, activation='relu', input_shape=(len(x_train[0]), )))
# mod2.add(Flatten())
# mod2.add(Dense(128, activation='relu'))
# mod2.add(Dropout(0.2))
mod2.add(Dense(32, activation='relu'))
mod2.add(Dropout(0.2))
mod2.add(Dense(1, activation='sigmoid'))
# compile the model
mod2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(mod2.summary())

mod2.fit(x_train, y_train, epochs=5)
# evaluate the model
loss, accuracy = mod2.evaluate(x_val, y_val)
print('\nAccuracy: %f' % (accuracy*100))