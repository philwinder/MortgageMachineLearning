import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Imputer, StandardScaler

from DataDescription import *


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

if __name__ == "__main__":


    X = getDefaultData("10000")
    X = X.append(getNonDefaultData(str(len(X))))

    # X = X.drop(["channel", "loan_purpose", "property_type", "occupancy_status", "original_loan_term", "number_of_units", "prepayment_penalty_flag", "first_time_homebuyer_flag"], axis=1)
    X = X[["credit_score", "hpi_at_origination", "dti", "default_flag"]]

    print("Data is equally matched over defaults and non-defaults. Size of data: " + str(len(X)))

    # X = one_hot_dataframe(X, ["channel", "loan_purpose", "property_type", "occupancy_status"], replace=True)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]

    # Target
    y = X["default_flag"]
    del X["default_flag"]

    # Replace NaNs with imputed values
    imp = Imputer(missing_values=0, strategy='mean', axis=0)
    imp.fit(X)
    X_imp = imp.transform(X)
    # Then scale
    X_norm, scaler = preprocess_data(X_imp)
    # Then put imputed values back to zero
    X_norm[(X == 0).as_matrix()] = 0

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [5, 10, 20, 50, 100, 200, 500, 1000]}]

    scores = ['precision']#, 'recall']

    # # Visualize data using PCA
    # pca = TruncatedSVD(n_components=2)
    # X_reduced = pca.fit_transform(X_train.as_matrix())
    #
    # ax = plt.figure(1)
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train.as_matrix(), s=50)
    # plt.title("PCA reduction (2d) of raw data (%dd)" %
    #              X_train.as_matrix().shape[1])
    # plt.xticks(())
    # plt.yticks(())
    # plt.tight_layout()
    # plt.show()


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()


        clf = GridSearchCV(ensemble.RandomForestClassifier(n_jobs=-1), tuned_parameters, cv=5, scoring='%s_weighted' % score)
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
        # plotImportances(clf.best_estimator_, X_train.columns, title="3. Feature importances after removal (Sample size = " + str(len(X)) + ")")
        # plotDescisionSurface(clf.best_estimator_, X_test, title="4. Decision boundary for features after removal (Sample size = " + str(len(X)) + ")")



# TODO: Plot Fscore vs number of features.
# TODO: Plot first two features vs classification.
# TODO: Plot 2 principal components vs classification