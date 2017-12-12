from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D
import sys
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def load_sent_embed(subj):
	pass


def load_data(files):
	'''
	loads the data and assigns the labels (binary classification with file 1 positive, file 2 negative)
	:param files: list of files for sentence data
	:return: list of sentences and their labels
	'''
	phrases = []
	file_len = []

	for fname in files:
		f = open(fname, 'rb')
		new_phrases = f.read().splitlines()
		phrases.extend(new_phrases)
		file_len.append(len(new_phrases))
		f.close()
	labels = []
	print(file_len)
	print(len(file_len))
	for i in range(len(file_len)):
		labels.extend(i for _ in range(file_len[i]))
	return phrases, labels


def token_to_embedding(word_index, embedding_index, dims):
	'''
	gives a lookup matrix for the rank of a word and its embedding
	[rank] ----> [ ... word embedding for (rank) ... ]

	:param word_index: dictionary for {word : rank }
	:param embedding_index: dictionary for embedding {word:word embedding}
	:param dims: dimensions of the matrix (number of columns)
	:return: a lookup matrix (list of list of word embeddings)
	'''
	vocab_size = len(word_index) + 1
	embedding_matrix = np.zeros((vocab_size, dims))

	for word, i in word_index.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


def load_embeddings(embedding_file):
	'''
	from a file return the dictionary (word:wordembedding)
	'''
	embeddings_index = dict()
	f = open(embedding_file)

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def full_embedding(tokenized_list_of_sent, embedding_matrix, cnn):
	'''
	take the embedding matrix and original list of sentences (as ranks not words)
	and return full embedding matrix of these sentences
	:param tokenized_list_of_sent:  list of sentences after being passed through the tokenizer (ranks not words)
	:param embedding_matrix: the word embedding matrix for each word
	:return: a matrix that is a list of sentences as word embeddings.
	'''
	full_embed_mat = []
	for sent in tokenized_list_of_sent:
		sentence_vector = []
		for word in sent:
			if cnn:
				sentence_vector.append(embedding_matrix[word])
			else:
				sentence_vector.extend(embedding_matrix[word])

		full_embed_mat.append(sentence_vector)

	return full_embed_mat


def data_to_embedding(phrases, cnn, sent_len=40, embedding_file='glove.6B.100d.txt', embedding_size=100):
	'''
	given a list of string sentences, it will transform them into list of sentence as word embedding
	:param phrases: list of string sentences
	:param sent_len: the maximum length you want to give to the sentence (after padding)
	:param embedding_file: where you get the embedding from
	:param embedding_size: size of the word embedding
	:return: a list of sentences as word embeddings
	'''

	tokenizer = Tokenizer(num_words=None,
						  filters='.,?!"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n',
						  lower=True,
						  split=" ",
						  char_level=False)
	tokenizer.fit_on_texts(phrases)
	# get embeddings from file
	embeddings_index = load_embeddings(embedding_file)
	# get matrix for word embeddings
	embedding_matrix = token_to_embedding(tokenizer.word_index, embeddings_index, embedding_size)
	# turn strings into ranks
	full_embedding_matrix = tokenizer.texts_to_sequences(phrases)
	# add to make sentences constant length
	full_embedding_matrix = sequence.pad_sequences(full_embedding_matrix, maxlen=sent_len, padding='post',
												   truncating='post')
	# turn into an embedding matrix for each sentence
	full_embedding_matrix = full_embedding(full_embedding_matrix, embedding_matrix, cnn)

	return full_embedding_matrix


def create_model(layer_1_neurons=128, layer_2_neurons=32, dropout=0.2, input_dim=4000):
	'''
	function to create a NN model.
	'''
	model = Sequential()
	# model.add(Embedding(input_dim=vocab_size, output_dim=dims, weights=[embedding_matrix], input_length=len(x_train[0]),
	# trainable=True))
	model.add(Dense(layer_1_neurons, activation='relu', input_dim=input_dim))
	model.add(Dropout(dropout))
	model.add(Dense(layer_2_neurons, activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	print(model.summary())
	return model

def create_cnn_model(dropout=0.2, input_dim=(40, 100)):
	model = Sequential()
	model.add(Conv1D(filters=48, kernel_size=4 ,strides=1, input_shape=input_dim, kernel_initializer= 'uniform', activation= 'relu')) 
	# model.add(Dropout(dropout))
	model.add(Conv1D(filters=16, kernel_size=4 ,strides=1, kernel_initializer= 'uniform', activation= 'relu')) 
	# model.add(Dropout(dropout))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	print(model.summary())
	return model


def main():
	cnn = True  # True for cnn, false for feed-forward NN
	sent = False  # True for sentence embeddings, false for word embeddings
	subj = True  # True for subjectivity dataset, false for polarity dataset
	if sent:
		data = load_sent_embed(subj)
	else:
		print('Loading data...')
		if subj:
			files = ["subjectivity-data/plot.tok.gt9.5000", "subjectivity-data/quote.tok.gt9.5000"]
		else:
			files = ["rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"]
		phrases, labels = load_data(files)
		print('Preprocessing data...')
		data = data_to_embedding(phrases, cnn, sent_len=40)

	# splitting into test (60%) validation(20%) and test (20%)
	seed = 7
	data = np.asarray(data, dtype=np.float32)
	print(data.shape)
	x_first_split, x_test, y_first_split, y_test = train_test_split(data, labels, test_size=0.3, random_state=seed)

	# --------------- simple way to make a model, train and test it ------------------
	if cnn:
		model_fn = create_cnn_model
		dim = (data.shape[1], data.shape[2])
	else:
		model_fn = create_model
		dim = len(data[0])

	print('Building the model...')
	model = KerasClassifier(build_fn=model_fn, epochs=3, dropout=0.2, input_dim=dim, verbose=0)

	# -------------- example cross validation -----------------------
	# kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
	# do cross validation on only the training and validation set (i.e. x_first_split)
	# results = cross_val_score(model, x_first_split, y_first_split, cv=kfold)
	# print("\naverage result:{0} , std: {1}".format(results.mean(),results.std()))

	# -------------- finally, produce predictions on test set ------
	model.fit(x_first_split, y_first_split)
	preds = model.predict(x_test)
	acc = accuracy_score(y_test, preds)
	print "accuracy: ", acc * 100

if __name__ == '__main__':
	main()

	