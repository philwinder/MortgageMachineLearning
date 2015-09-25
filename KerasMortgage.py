from __future__ import absolute_import
from __future__ import print_function
from datetime import datetime

from PIL import Image
from keras.models import Sequential
from keras.layers.core import AutoEncoder, Dense, Activation, Dropout
from keras.layers import containers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, Imputer

from DataDescription import getDefaultData, getNonDefaultData
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

RANDOM_SEED = 859438905

def printImportances(forest, features):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(importances)):
        print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

def plotImportances(forest, features, title=""):
    # Plot the feature importances of the forest
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    fig = plt.figure()                      # initialize figure
    ax = fig.add_axes([0.1, 0.4, 0.8, 0.5]) # add axis
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(importances)), features[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.savefig("plots/" + title + '.png',bbox_inches='tight')
    plt.close()


def one_hot_dataframe(data, cols, replace=False):
    vecData = pd.get_dummies(data, columns=cols)
    return vecData

def plotDescisionSurface(forest, X, title=""):
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Calculate mean threshold
    val = [0] * len(forest.estimators_)
    i = 0;
    for tree_in_forest in forest.estimators_:
        val = val + tree_in_forest.tree_.threshold
        i = i + 1
    val = val / i

    for plot_idx in indices[range(9)]:
        plt.subplot(3, 3, plot_idx)
        plt.title(X.columns[plot_idx])

        X.ix[:, plot_idx].hist(color='k')


    plt.suptitle(title)
    plt.axis("tight")
    plt.savefig("plots/" + title + '.png',bbox_inches='tight')
    plt.close()

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

X = getDefaultData("1000")
X = X.append(getNonDefaultData(str(len(X))))


X = X.drop(["channel", "loan_purpose", "property_type", "occupancy_status", "original_loan_term", "number_of_units", "prepayment_penalty_flag", "first_time_homebuyer_flag"], axis=1)
# X = X[["credit_score", "hpi_at_origination", "dti", "default_flag"]]

print("Data is equally matched over defaults and non-defaults. Size of data: " + str(len(X)))

# X = one_hot_dataframe(X, ["channel", "loan_purpose", "property_type", "occupancy_status"], replace=True)

# Shuffle
idx = np.random.permutation(len(X))
X = X.iloc[idx]

# Target
y = X["default_flag"]
del X["default_flag"]

## END OF PANDAS DF


# Replace NaNs with imputed values
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(X)
X_imp = imp.transform(X)
# Then scale
X_norm, scaler = preprocess_data(X_imp)
# Then put imputed values back to zero
X_norm[(X == 0).as_matrix()] = 0


# Split the dataset in two equal parts
X_train, X_test, Y_train, y_test = train_test_split(
    X_norm, y, test_size=0.5, random_state=0)

nb_classes = 2
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size = 128
activation = 'sigmoid'

input_dim = X_train.shape[1]
hidden_dim = 2

nb_epoch = 20

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)


##########################
# dense model test       #
##########################

print("Training classical fully connected layer for classification")
model_classical = Sequential()
model_classical.add(Dense(input_dim, 2, activation=activation))
model_classical.add(Dropout(0.3))
model_classical.add(Activation('softmax'))
model_classical.compile(loss='binary_crossentropy', optimizer='adam')
model_classical.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
classical_score = model_classical.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
print('classical_score:', classical_score)

y_true, y_pred = y_test, model_classical.predict_classes(X_test)
print(classification_report(y_true, y_pred))

##########################
# autoencoder model test #
##########################
# plotExamples(X_train, number = 25, title="Original_")

print("Training AutoEncoder for feature viz")

# AutoEncoder for feature visualization
autoencoder = Sequential()
encoder = containers.Sequential([Dense(input_dim, hidden_dim), Dropout(0.3), Activation(activation)])
decoder = containers.Sequential([Dense(hidden_dim, input_dim, activation=activation)])
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

autoencoder.compile(loss='mse', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
# plotExamples(autoencoder.predict(X_train), number = 25, title="Reproduction_")
vals = getActivationsLayer0(autoencoder.layers[0].encoder.layers)[0].T
# plotExamples(vals, number = hidden_dim, title="Neuron_features_")

print("Training AutoEncoder for TSNE and classification")

# AutoEncoder for TSNE and classification
autoencoder = Sequential()
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))

autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))

# Do an inference pass
prefilter_train = autoencoder.predict(X_train, verbose=0)
prefilter_test = autoencoder.predict(X_test, verbose=0)
print("prefilter_train: ", prefilter_train.shape)
print("prefilter_test: ", prefilter_test.shape)

y_true, y_pred = y_test, autoencoder.predict_classes(X_test)
print(classification_report(y_true, y_pred))


# #
# # print("Performing TSNE")
# # model = TSNE(n_components=2, random_state=0, init="pca")
# # toPlot = model.fit_transform(prefilter_test)
# # plotTSNE(toPlot, y_test, nb_classes, "t-SNE embedding for AutoEncoded output")
# #
# #
# # # Visualize result using PCA
# # pca = TruncatedSVD(n_components=2)
# # X_reduced = pca.fit_transform(X_train.as_matrix())
# # plotTSNE(X_reduced, y_test, nb_classes, "PCA for AutoEncoded output")
#
#
# print("Classifying and comparing")
# # Classify results from Autoencoder
# print("Building classical fully connected layer for classification")
# model = Sequential()
# model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))
#
# model.add(Activation('softmax'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(prefilter_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(prefilter_test, Y_test))
#
# score = model.evaluate(prefilter_test, Y_test, verbose=0, show_accuracy=True)
# print('\nscore:', score)
#
# print('Loss change:', 100*(score[0] - classical_score[0])/classical_score[0], '%')
# print('Accuracy change:', 100*(score[1] - classical_score[1])/classical_score[1], '%')
#
# y_true, y_pred = y_test, model.predict_classes(prefilter_test)
# print(classification_report(y_true, y_pred))