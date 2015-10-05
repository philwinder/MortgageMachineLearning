from __future__ import absolute_import
from __future__ import print_function
from datetime import datetime

from PIL import Image
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import AutoEncoder, Dense, Activation, Dropout
from keras.layers import containers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, Imputer
import seaborn as sns
sns.set(style='white', palette='Set2')

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
    plt.figure()
    y_unique = np.unique(labels)
    cmap = sns.color_palette("Set2", n_colors=len(y_unique))
    for i in range(0, len(y_unique)):
        plt.scatter(toPlot[labels.as_matrix()==y_unique[i], 0], toPlot[labels.as_matrix()==y_unique[i], 1], c=cmap[i], label=y_unique[i], s=50)
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.savefig("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png', bbox_inches='tight')

def plot3DScatter(toPlot, labels, nb_classes, title = ""):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = np.min(toPlot, 0), np.max(toPlot, 0)
    toPlot = (toPlot - x_min) / (x_max - x_min)
    print(toPlot.shape)
    y_unique = np.unique(labels)
    cmap = sns.color_palette("Set2", n_colors=len(y_unique))
    for i in range(0, len(y_unique)):
        ax.scatter(toPlot[labels.as_matrix()==y_unique[i], 0], toPlot[labels.as_matrix()==y_unique[i], 1], toPlot[labels.as_matrix()==y_unique[i], 2], c=cmap[i], label=y_unique[i], s=50)
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    # sns.despine(left=True, bottom=True)
    plt.savefig("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png', bbox_inches='tight')

RANDOM_SEED = 859438905

def one_hot_dataframe(data, cols, replace=False):
    vecData = pd.get_dummies(data, columns=cols)
    return vecData

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

X = getDefaultData("10000")
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

# Hot one encode
X = one_hot_dataframe(X, ["number_of_borrowers"])

# Replace NaNs with imputed values
imp = Imputer(missing_values=0, strategy='mean', axis=0, copy=False)
X_imp = imp.fit_transform(X)


# Then scale
X_norm, scaler = preprocess_data(X_imp)
# Then put imputed values back to zero
X_norm[X.as_matrix()==0] = 0


# Split the dataset in two equal parts
X_train, X_test, Y_train, y_test = train_test_split(
    X_norm, y, test_size=0.1, random_state=0)

nb_classes = 2
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size = 128
activation = 'sigmoid'

input_dim = X_train.shape[1]
hidden_dim = 10
final_dim = 3

nb_epoch = 500

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

# print("Performing TSNE")
# model = TSNE(n_components=2, random_state=0, init="pca")
# toPlot = model.fit_transform(X_test)
# plotTSNE(toPlot, y_test, nb_classes, "1_t-SNE embedding for AutoEncoded output")


# Visualize result using PCA
print("Performing PCA")
pca = TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_test)
plotTSNE(X_reduced, y_test, nb_classes, "2_PCA for AutoEncoded output")

##########################
# dense model test       #
##########################

# print("Training classical fully connected layer for classification")
# model_classical = Sequential()
# model_classical.add(Dense(input_dim, 2, activation=activation))
# model_classical.add(Dropout(0.3))
# model_classical.add(Activation('softmax'))
# model_classical.compile(loss='binary_crossentropy', optimizer='adadelta')
# model_classical.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
# classical_score = model_classical.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
# print('classical_score:', classical_score)
#
# y_true, y_pred = y_test, model_classical.predict_classes(X_test)
# print(classification_report(y_true, y_pred))

##########################
# autoencoder model test #
##########################
# plotExamples(X_train, number = 25, title="Original_")

# print("Training AutoEncoder for feature viz")

# AutoEncoder for feature visualization
encoder = containers.Sequential([Dense(input_dim, hidden_dim, activation=activation), Dense(hidden_dim, final_dim, activation=activation)])
decoder = containers.Sequential([Dense(final_dim, hidden_dim, activation=activation), Dense(hidden_dim, input_dim, activation=activation)])

# autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

# autoencoder.compile(loss='mse', optimizer='adadelta')
# # Do NOT use validation data with return output_reconstruction=True
# autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
# # plotExamples(autoencoder.predict(X_train), number = 25, title="Reproduction_")
# vals = getActivationsLayer0(autoencoder.layers[0].encoder.layers)[0].T
# # plotExamples(vals, number = hidden_dim, title="Neuron_features_")

print("Training AutoEncoder for TSNE and classification")

# AutoEncoder for TSNE and classification
autoencoder = Sequential()
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))

autoencoder.compile(loss='mean_squared_error', optimizer='adam')
# Do NOT use validation data with return output_reconstruction=True
autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1)


# Do an inference pass
prefilter_train = autoencoder.predict(X_train, verbose=0)
prefilter_test = autoencoder.predict(X_test, verbose=0)
print("prefilter_train: ", prefilter_train.shape)
print("prefilter_test: ", prefilter_test.shape)


print("Performing TSNE")
model = TSNE(n_components=2, random_state=0, init="pca")
toPlot = model.fit_transform(prefilter_test)
plotTSNE(toPlot, y_test, nb_classes, "3_t-SNE embedding for AutoEncoded output")

plot3DScatter(prefilter_test, y_test, nb_classes, "5_3D_Raw_AutoEncoded output")

# Visualize result using PCA
pca = TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(prefilter_test)
plotTSNE(X_reduced, y_test, nb_classes, "4_PCA for AutoEncoded output")

#
# print("Classifying and comparing")
# # Classify results from Autoencoder
# model = Sequential()
# model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))
#
# model.add(Activation('softmax'))
#
# model.compile(loss='binary_crossentropy', optimizer='adadelta')
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