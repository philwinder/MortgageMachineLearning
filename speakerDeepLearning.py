from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import seaborn as sns
sns.set(style='ticks', palette='Set2')

import gzip

from pylab import *

np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    gzip_file(fname)
    print("Wrote submission to file {}.".format(fname))

def gzip_file(file):
    with open(file, "rb") as file_in:
        # Open output file.
        with gzip.open(file + ".tar.gz", "wb") as file_out:
            # Write output.
            file_out.writelines(file_in)




print("Loading data...")
X, labels = load_data('both.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

# X_test, ids = load_data('../raw/test.csv', train=False)
# X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

testVal = 512

model = Sequential()
model.add(Dense(dims, testVal, init='glorot_uniform'))
model.add(PReLU((testVal,)))
model.add(BatchNormalization((testVal,)))
model.add(Dropout(0.2))

model.add(Dense(testVal, testVal, init='glorot_uniform'))
model.add(PReLU((testVal,)))
model.add(BatchNormalization((testVal,)))
model.add(Dropout(0.2))

model.add(Dense(testVal, testVal, init='glorot_uniform'))
model.add(PReLU((testVal,)))
model.add(BatchNormalization((testVal,)))
model.add(Dropout(0.2))

model.add(Dense(testVal, nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adadelta")

print("Training model...")

model.fit(X, y, nb_epoch=100, batch_size=5, validation_split=0.01)

print("Generating results...")

proba = model.predict_proba(X)
print(model.evaluate(X,y, show_accuracy=True))
print(proba)
print(model.predict_classes(X), labels)
print(classification_report(labels-1, model.predict_classes(X)))

plt.figure()
plt.scatter(proba[:,0], proba[:,1])

savefig('Classified.png',bbox_inches='tight')


print("Unshuffled")
X, labels = load_data('both.csv', train=False)
X = scaler.transform(X)
y, encoder = preprocess_labels(labels)
newProba = model.predict_proba(X)
print(model.evaluate(X,y, show_accuracy=True))

print(newProba[2])
print(newProba[11])
print(newProba[21])
print("1")
print(newProba[0:9])
print("2")
print(newProba[10:19])
print("3")
print(newProba[20:-1])
print(labels)
print(model.predict_classes(X))


from sklearn.manifold import TSNE
model = TSNE(n_components=nb_classes, random_state=0, init='pca')
toPlot = model.fit_transform(newProba)


title = "t-SNE embedding of the spectrograms"

x_min, x_max = np.min(toPlot, 0), np.max(toPlot, 0)
toPlot = (toPlot - x_min) / (x_max - x_min)
print(toPlot.shape)

labelsName = ["bob", "steve", "dave"]

cmap = sns.color_palette("Set2", n_colors=3)

plt.figure()
for i in range(toPlot.shape[0]):
    plt.text(toPlot[i, 0], toPlot[i, 1], labelsName[int(labels[i])-1] + "_" + str(i),
             color=cmap[int(labels[i])-1],
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([]), plt.yticks([])
sns.despine()
savefig('ClassifiedTSNE.png',bbox_inches='tight')

