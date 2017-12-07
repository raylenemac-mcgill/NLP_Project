import sys
sys.path.append("\Users\Raylene\Documents\Courses\Fall 2017 Courses\COMP550\Final Project\New Proj\NLP_Project\skip-thoughts")

import skipthoughts
import cPickle


# f = open("rt-polaritydata/rt-polarity.neg", "rb")
f = open("subjectivity-data/plot.tok.gt9.5000", "rb")
phrases = f.read().splitlines()[:5]
len_class0 = len(phrases)
# f = open("rt-polaritydata/rt-polarity.pos", "rb")
f = open("subjectivity-data/quote.tok.gt9.5000", "rb")
phrases.extend(f.read().splitlines()[:5])
len_class1 = len(phrases) - len_class0

phrases = [s.decode("ascii", errors="ignore").encode() for s in phrases]
print(phrases[:3])
print(len(phrases))

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
vectors = encoder.encode(phrases)

print(len(vectors))
print(vectors[0])
print(len(vectors[0]))

with open(r"vectors_5.pickle", "wb") as output_file:
	cPickle.dump(vectors, output_file)
