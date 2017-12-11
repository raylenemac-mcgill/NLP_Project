from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout
import sys
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


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


def full_embedding(tokenized_list_of_sent, embedding_matrix):
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
			sentence_vector.extend(embedding_matrix[word])

		full_embed_mat.append(sentence_vector)

	return full_embed_mat


def data_to_embedding(phrases, sent_len=40, embedding_file='glove.6B.100d.txt', embedding_size=100):
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
	full_embedding_matrix = full_embedding(full_embedding_matrix, embedding_matrix)

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
	return model


def main():
	# files should be a list of string file names
	files = ["rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"]
	print('Loading data...')
	phrases, labels = load_data(files)
	print('Preprocessing data...')
	data = data_to_embedding(phrases, sent_len=51)

	# splitting into test (60%) validation(20%) and test (20%)
	x_first_split, x_test, y_first_split, y_test = train_test_split(data, labels, test_size=0.2)
	x_train, x_val, y_train, y_val = train_test_split(x_first_split, y_first_split, test_size=0.2)

	# --------------- simple way to make a model, train and test it ------------------
	print('Training the model...')
	model = KerasClassifier(build_fn=create_model, epochs=4, dropout=0.2, input_dim=5100, verbose=0)
	model.fit(x_train, y_train)
	# -------------- example cross validation -----------------------
	seed = 7
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	# do cross validation on only the training and validation set (i.e. x_first_split)
	results = cross_val_score(model, x_first_split, y_first_split, cv=kfold)
	print("average result:{0} , std: {1}".format(results.mean(),results.std()))

	# -------------- finally, produce predictions on test set ------
	preds = model.predict(x_test)
	acc = accuracy_score(y_test, preds)
	print(acc * 100)

if __name__ == '__main__':
	main()

	# ---------------- THE PREVIOUS WAY OF DOING CROSS VALIDATION, CREATING THE MODEL EACH TIME -----------

	# # the data has to be an np array to be split in the way below X[train} etc. it will give an error otherwise
	# X = np.array(data)
	# Y = np.array(labels)
	# # CROSS-VALIDATION
	# seed = 7
	# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	# cvscores = []
	# for train, test in kfold.split(X, Y):
	#
	# 	x_train, x_test = X[train], X[test]
	# 	y_train, y_test = Y[train], Y[test]
	# 	#create model
	# 	print('Creating the model...')
	# 	model = KerasClassifier(build_fn=create_model,input_dim=5100,layer_1_neurons=layer_number,epochs=4, verbose=0)
	# 	model.fit(x_train, y_train, epochs=4)
	# 	scores = model.evaluate(x_test, y_test, verbose=0)
	# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	# 	cvscores.append(scores[1] * 100)
	# print("%.2f%% (+/- %.2f%%)" % (np.mean(np.array(cvscores)), np.std(cvscores)))
	# ------------------------------------------------------------------------------------------------------

	# ------------ manually doing grid search, super slow. The other way didn't work with me --------------
	# print('Training the model...')
	# layer_1_input = [128, 500, 1000, 2000]
	# layer_2_input = [32, 128, 500, 1000]
	# seed = 7
	# for layer_number in layer_1_input:
	# 	for layer_2_number in layer_2_input:
	# 		model = KerasClassifier(build_fn=create_model, input_dim=5100, layer_1_neurons=layer_number, layer_2_neurons=layer_2_number, epochs=4,
	# 								verbose=0)
	# 		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	# 		# cross validation on only the test and validation set
	# 		results = cross_val_score(model, x_first_split, y_first_split, cv=kfold)
	# 		print("average accuracy: {0} std: {1}, for layers with H1:{2},H2:{3} neurons".format(results.mean(),
	# 																							 results.std(),
	# 																							 layer_number,
	# 																							 layer_2_number))
	# sys.exit()