import cPickle
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, Input, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split

with open(r"vectors.pickle", "rb") as input_file:
	vecs = cPickle.load(input_file)

print(len(vecs))

len_class0, len_class1 = 300, 300
labels = np.zeros(len_class0 + len_class1)
labels[:len_class1] = 1.0
labels[len_class1:] = 0.0

x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.33)

mod2 = Sequential()
mod2.add(Dense(128, input_shape=(len(x_train[0]), )))
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
loss, accuracy = mod2.evaluate(x_test, y_test)
print('\nAccuracy: %f' % (accuracy*100))