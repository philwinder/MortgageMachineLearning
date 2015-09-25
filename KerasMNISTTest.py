from __future__ import absolute_import
from __future__ import print_function
from datetime import datetime

from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import AutoEncoder, Dense, Activation, Dropout
from keras.layers import containers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import tile_raster_images

def plotExamples(data, number = 9, title=""):
    title = title + "Example_data"
    print("Plotting " + title)
    image_size = np.floor(np.sqrt(data.shape[1]))
    subplot_size = int(np.floor(np.sqrt(number)))
    raster = tile_raster_images(
        X=data,
        img_shape=(image_size, image_size), tile_shape=(subplot_size, subplot_size),
        tile_spacing=(1, 1))
    image = Image.fromarray(raster)
    image.save("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png')

def getActivationsLayer0(layers, requested_layer=0):
    top_layer = layers[0].get_weights()[0]
    bot_layer = layers[requested_layer].get_weights()[0]
    input_dim = top_layer.shape[0]
    bot_dim = bot_layer.shape[1]
    y = np.zeros([input_dim, bot_dim])
    W = top_layer
    for i in range(0, input_dim):
        w_norm = np.sqrt(sum(np.square(W[i, :])))
        y[i, :] = W[i, :] / w_norm
    return y, input_dim, bot_dim

def plotTSNE(toPlot, labels, nb_classes, title = ""):
    x_min, x_max = np.min(toPlot, 0), np.max(toPlot, 0)
    toPlot = (toPlot - x_min) / (x_max - x_min)
    print(toPlot.shape)
    cm = plt.cm.Set1(255 * np.arange(0, nb_classes) / nb_classes)
    plt.figure()
    for i in range(toPlot.shape[0]):
        plt.text(toPlot[i, 0], toPlot[i, 1], str(labels[i]),
                 color=cm[labels[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png', bbox_inches='tight')

nb_classes = 10
batch_size = 128
activation = 'sigmoid'

input_dim = 784
hidden_dim = 250

nb_epoch = 20
max_train_samples = 5000
max_test_samples = 1000

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, input_dim)[:max_train_samples]
X_test = X_test.reshape(10000, input_dim)[:max_test_samples]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
y_train = y_train[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]
y_test = y_test[:max_test_samples]

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


##########################
# dense model test       #
##########################

print("Training classical fully connected layer for classification")
model_classical = Sequential()
model_classical.add(Dense(input_dim, 10, activation=activation))
model_classical.add(Activation('softmax'))
model_classical.compile(loss='categorical_crossentropy', optimizer='adam')
model_classical.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
classical_score = model_classical.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
print('classical_score:', classical_score)

##########################
# autoencoder model test #
##########################
plotExamples(X_train, number = 25, title="Original_")

print("Training AutoEncoder for feature viz")

# AutoEncoder for feature visualization
autoencoder = Sequential()
encoder = containers.Sequential([Dense(input_dim, hidden_dim), Dropout(0.3), Activation(activation)])
decoder = containers.Sequential([Dense(hidden_dim, input_dim, activation=activation)])
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

autoencoder.compile(loss='mean_squared_error', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
plotExamples(autoencoder.predict(X_train), number = 25, title="Reproduction_")
vals = getActivationsLayer0(autoencoder.layers[0].encoder.layers)[0].T
plotExamples(vals, number = hidden_dim, title="Neuron_features_")

print("Training AutoEncoder for TSNE and classification")

# AutoEncoder for TSNE and classification
autoencoder = Sequential()
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))

autoencoder.compile(loss='mean_squared_error', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)

# Do an inference pass
prefilter_train = autoencoder.predict(X_train, verbose=0)
prefilter_test = autoencoder.predict(X_test, verbose=0)
print("prefilter_train: ", prefilter_train.shape)
print("prefilter_test: ", prefilter_test.shape)



print("Performing TSNE")
model = TSNE(n_components=2, random_state=0, init="pca")
toPlot = model.fit_transform(prefilter_test)
plotTSNE(toPlot, y_test, nb_classes, "t-SNE embedding for AutoEncoded output")



print("Classifying and comparing")
# Classify results from Autoencoder
print("Building classical fully connected layer for classification")
model = Sequential()
model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(prefilter_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(prefilter_test, Y_test))

score = model.evaluate(prefilter_test, Y_test, verbose=0, show_accuracy=True)
print('\nscore:', score)

print('Loss change:', 100*(score[0] - classical_score[0])/classical_score[0], '%')
print('Accuracy change:', 100*(score[1] - classical_score[1])/classical_score[1], '%')

