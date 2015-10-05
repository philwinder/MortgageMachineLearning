from datetime import datetime
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Imputer, StandardScaler

from DataDescription import *
import seaborn as sns
sns.set(style='ticks', palette='Set2')

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
    cmap = sns.color_palette("Set2", n_colors=2)
    plt.bar(range(len(importances)), importances[indices], color=cmap[0], ecolor=cmap[1],  yerr=std[indices], align="center")
    plt.xticks(range(len(importances)), features[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylim([0, 0.5])
    sns.despine()
    plt.savefig("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png',bbox_inches='tight')
    plt.close()

def one_hot_dataframe(data, cols, replace=False):
    vecData = pd.get_dummies(data, columns=cols)
    return vecData

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pca(X):
    pca = TruncatedSVD(n_components=2)
    return pca.fit_transform(X)

def plotScatter(X, y, title=""):
    # Visualize data using PCA
    ax = plt.figure(1)
    y_mat = y.as_matrix()
    y_unique = np.unique(y_mat)
    cmap = sns.color_palette("Set2", n_colors=len(y_unique))
    for i in range(0, len(y_unique)):
        plt.scatter(X[y_mat==y_unique[i], 0], X[y_mat==y_unique[i], 1], c=cmap[i], label=y_unique[i], s=50)
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.savefig("plots/" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + title + '.png', bbox_inches='tight')
    plt.close()

def plotTSNE(toPlot, labels, nb_classes, title = ""):
    print("Plotting TSNE")
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

if __name__ == "__main__":

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
    imp = Imputer(missing_values=0, strategy='mean', axis=0)
    X_imp = imp.fit_transform(X)


    # Then scale
    X_norm, scaler = preprocess_data(X_imp)
    # Then put imputed values back to zero
    X_norm[X.as_matrix()==0] = 0


    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.1, random_state=RANDOM_SEED)

    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [5, 10, 20, 50, 100, 200]}]

    scores = ['precision']#, 'recall']

    print("Performing PCA")
    X_pca = pca(X_train)
    plotScatter(X_pca[:1000], y_train[:1000], title="1_PCA reduction (2d) of raw data (%dd)" % X_train.shape[1])

    print("Performing TSNE")
    model = TSNE(n_components=2, random_state=RANDOM_SEED, init="pca")
    toPlot = model.fit_transform(X_train[:1000])
    plotTSNE(toPlot, y_test[:1000], 2, "2_t-SNE embedding for AutoEncoded output")

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()


        clf = GridSearchCV(ensemble.RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, min_samples_leaf=10), tuned_parameters, cv=5, scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()


        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

        # printImportances(clf.best_estimator_, X_train.columns)
        plotImportances(clf.best_estimator_, X.columns, title="3. Feature importances after removal (Sample size = " + str(len(X)) + ")")


# TODO: Plot Fscore vs number of features.
# TODO: Plot first two features vs classification.
# TODO: Plot 2 principal components vs classification