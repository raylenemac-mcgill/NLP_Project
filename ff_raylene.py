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

import keras


def load_data(files):
	'''

	:param files: list of files
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


def get_train_test(phrases, labels, sent_len=40, embedding_file='glove.6B.100d.txt',embedding_size = 100):
	'''

	:param phrases: list of string sentences
	:param labels: the labels for each string sentence from phrases ( make sure they are aligned, i.e. have same index)
	:param sent_len: how big you want your sentence length to be (truncation or addition of 0's)
	:param embedding_file: text file (in same directory) for specifying a word embedding, otherwise just use standford's
	:param dims: the dimension of your word embedding ( for the stanford word embedding 100)
	:return: x_train ( a matrix of word embedding for a sentence), x_test, y_train, y_test
	'''

	x_train, x_test, y_train, y_test = train_test_split(phrases, labels, test_size=0.3)

	tokenizer = Tokenizer(num_words=None,
						  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
						  lower=True,
						  split=" ",
						  char_level=False)
	tokenizer.fit_on_texts(phrases)
	embeddings_index = load_embeddings(embedding_file)
	embedding_matrix = token_to_embedding(tokenizer.word_index, embeddings_index, embedding_size)

	for x in [x_train,x_test]:
		x =  tokenizer.texts_to_sequences(x)
		x = sequence.pad_sequences(x, maxlen=sent_len, padding='post', truncating='post')
		x = full_embedding(x_train, tokenizer.word_index, embedding_matrix)

	# #x_train =
	# x_test = tokenizer.texts_to_sequences(x_test)
	#
	# x_train =
	# x_test = sequence.pad_sequences(x_test, maxlen=sent_len, padding='post', truncating='post')
	# x = full_embedding(x_train, tokenizer.word_index, embedding_matrix)
	# x2 = full_embedding(x_test, tokenizer.word_index, embedding_matrix)

	return x_train, x_test, y_train, y_test


def token_to_embedding(token_dict, embedding_index, dims):
	vocab_size = len(token_dict) + 1
	embedding_matrix = np.zeros((vocab_size, dims))

	for word, i in token_dict.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


def load_embeddings(embedding_file):
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

	#print(tokenized_list_of_sent[:5])
	full_embed_mat=[]
	counter = 0
	for sent in tokenized_list_of_sent:
		#print("NEW SENTENCE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		vector =[]
		for word in sent:
			#print("----------------------------------------------------------------")

			#print(word)
			#print(embedding_matrix[word])
			#print(len(embedding_matrix[word]))
			vector.extend(embedding_matrix[word])

		counter +=1
		full_embed_mat.append(vector)
		#print(full_embed_mat)
	#print(full_embed_mat[-1])
	return full_embed_mat


def create_model():
	model = Sequential()
	# model.add(Embedding(input_dim=vocab_size, output_dim=dims, weights=[embedding_matrix], input_length=len(x_train[0]),
	# trainable=True))
	model.add(Dense(128, activation='relu', input_dim=4000))
	# model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	return model


def main():
	# files should be a list of string file names
	files = ["rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"]
	print('Loading data...')
	phrases, labels = load_data(files)
	print('Pre processing data...')

	sent_len=40
	embedding_file = 'glove.6B.100d.txt'
	embedding_size = 100

	# get the embeddings from a file
	# x_train, x_test, y_train, y_test = train_test_split(phrases, labels, test_size=0.3)

	tokenizer = Tokenizer(num_words=None,
						  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
						  lower=True,
						  split=" ",
						  char_level=False)
	tokenizer.fit_on_texts(phrases)
	embeddings_index = load_embeddings(embedding_file)
	embedding_matrix = token_to_embedding(tokenizer.word_index, embeddings_index, embedding_size)

	# x_train = tokenizer.texts_to_sequences(x_train)
	# x_test = tokenizer.texts_to_sequences(x_test)
	#
	# x_train = sequence.pad_sequences(x_train, maxlen=sent_len, padding='post', truncating='post')
	# x_test = sequence.pad_sequences(x_test, maxlen=sent_len, padding='post', truncating='post')
	#
	# x_train = full_embedding(x_train, embedding_matrix)
	# x_test = full_embedding(x_test, embedding_matrix)

	# tokenized_sentences=[x_train,x_test]
	# x_train = get_embeddings(x_train)
	# i = 0
	# x = tokenizer.texts_to_sequences(x)
	# # 	x = sequence.pad_sequences(x, maxlen=sent_len, padding='post', truncating='post')
	# # 	x = full_embedding(x, embedding_matrix)

	dims = 100

	# print('Creating the model...')
	# print(keras.backend.image_data_format())
	# model = Sequential()
	# #model.add(Embedding(input_dim=vocab_size, output_dim=dims, weights=[embedding_matrix], input_length=len(x_train[0]),
	# 					#trainable=True))
	seed = 7

	X = np.array(phrases)
	Y = np.array(labels)

	# CROSS-VALIDATION
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	cvscores = []
	for train, test in kfold.split(X, Y):

		x_train, x_test = X[train], X[test]
		y_train, y_test = Y[train], Y[test]
		# can be put outside .....
		x_train = tokenizer.texts_to_sequences(x_train)
		x_test = tokenizer.texts_to_sequences(x_test)

		x_train = sequence.pad_sequences(x_train, maxlen=sent_len, padding='post', truncating='post')
		x_test = sequence.pad_sequences(x_test, maxlen=sent_len, padding='post', truncating='post')

		x_train = full_embedding(x_train, embedding_matrix)
		x_test = full_embedding(x_test, embedding_matrix)

		model = KerasClassifier(build_fn=create_model, verbose=0)
		epochs = [2, 3, 4]
		param_grid = dict(epochs=epochs)
		grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
		grid_result = grid.fit(x_train, y_train)
		# summarize results
		print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))
		#create model
		print('Creating the model...')
		model = Sequential()
		# model.add(Embedding(input_dim=vocab_size, output_dim=dims, weights=[embedding_matrix], input_length=len(x_train[0]),
		# trainable=True))
		model.add(Dense(128, activation='relu', input_dim=4000))
		# model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(32, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(1, activation='sigmoid'))
		# compile the model
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
		model.fit(x_train, y_train, epochs=4)
		scores = model.evaluate(x_test, y_test, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(np.array(cvscores)), np.std(cvscores)))

if __name__ == '__main__':
	main()


# models = [LinearSVC(), LogisticRegression(), MultinomialNB()]
# 		names = ['SVM', "Logistic Reg"]
# 		for name, model in zip(names, models):
# 			model.fit(x_train, y_train)
# 			preds = model.predict(x_test)
# 			acc = accuracy_score(y_test, preds)
# 			model_score[name].append(acc)
# 			print("Model {} got accuracy {}".format(name, acc))
# 	print("%.2f%% (+/- %.2f%%)" % (np.mean(np.array(model_score[x])), np.std(model_score[x])) for x in model_score.keys())
# svm = [0.629213483146, 0.679174484053, 0.642589118199, 0.648217636023, 0.666041275797, 0.630393996248,
	# 			   0.627579737336, 0.663227016886, 0.653846153846, 0.631332082552]
	# logreg = [0.642589118199,
	# 				  0.659176029963,
	# 				  0.705440900563,
	# 				  0.65478424015,
	# 				  0.693245778612,
	# 				  0.683864915572,
	# 				  0.662288930582,
	# 				  0.68105065666,
	# 				  0.653846153846,
	# 				  0.65009380863]
	#
	# print("%.2f%% (+/- %.2f%%)" % (np.mean(svm), np.std(svm)))
	# print("%.2f%% (+/- %.2f%%)" % (np.mean(np.array(logreg)), np.std(logreg)))