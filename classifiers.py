import cPickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

with open(r"vectors.pickle", "rb") as input_file:
	vecs = cPickle.load(input_file)

# print(vecs)
print(len(vecs))
# print(len(vecs[0]))

len_class0, len_class1 = 300, 300
# labels = [int(0.0) for _ in range(len_class0)] + [int(1.0) for _ in range(len_class1)]
# print(labels)
labels = np.zeros(len_class0 + len_class1)
labels[:len_class1] = 1.0
labels[len_class1:] = 0.0

# x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.33)

# # Train classifier
# clf = LogisticRegression()
# clf.fit(x_train, y_train)
# acc = clf.score(x_test, y_test)
# print(acc)


k=10
seed=1234
scan = [2**t for t in range(0,9,1)]
npts = len(vecs)
# kf = KFold(npts, n_folds=k, random_state=seed)
kf = KFold(n_splits=k, random_state=seed, shuffle=True)
scores = []
for train_index, test_index in kf.split(vecs):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # sys.exit()
    # Split data
    X_train = vecs[train_index]
    y_train = labels[train_index]
    X_test = vecs[test_index]
    y_test = labels[test_index]

    scanscores = []
    for s in scan:

        # Inner KFold
        innerkf = KFold(n_splits=k, random_state=seed+1)  # KFold(len(X_train), n_folds=k, random_state=seed+1)
        innerscores = []
        for innertrain, innertest in innerkf.split(X_train):
    
            # Split data
            X_innertrain = X_train[innertrain]
            y_innertrain = y_train[innertrain]
            X_innertest = X_train[innertest]
            y_innertest = y_train[innertest]

            # Train classifier
            clf = LogisticRegression(C=s)    # C is a regularization parameter; we are doing CV to find the best value for it
            clf.fit(X_innertrain, y_innertrain)
            acc = clf.score(X_innertest, y_innertest)
            innerscores.append(acc)
            # print (s, acc)

        # Append mean score
        scanscores.append(np.mean(innerscores))

    # Get the index of the best score
    s_ind = np.argmax(scanscores)
    s = scan[s_ind]
    # print scanscores
    print ("best s: ", s)
   
    # Train classifier
    clf = LogisticRegression(C=s)
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    scores.append(acc)
print scores
    # [0.94999999999999996, 0.8833333333333333, 0.8666666666666667, 0.91666666666666663, 
    # 0.93333333333333335, 0.90000000000000002, 0.94999999999999996, 0.90000000000000002, 0.90000000000000002, 0.91666666666666663]  s = 32, 32, 64, 256

# return scores