import sys

from google.colab import drive
drive.mount('/content/drive')
sys.path += ['/content/drive/My Drive/ComSciM146/'] 

from nutil import *

import math
import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        self.probabilities_ = {0: 0, 1: 0}
        for cls in y:
          self.probabilities_[cls] += 1
          
        total = sum(self.probabilities_.values())

        self.probabilities_[0] = self.probabilities_[0]/total
        self.probabilities_[1] = self.probabilities_[1]/total
        

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        n,d = X.shape
        y = np.random.choice(2, n, p = [self.probabilities_[0], self.probabilities_[1]])


        return y



def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    ncol = 3
    nrow = d // ncol + 1
    for i in range(d) :
        fig.add_subplot (nrow,ncol,i+1)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() 
        plt.show()

    return data, bins, align, labels

def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
        f1_score    -- float, test "micro" averaged f1 score
    """

    # compute cross-validation error using StratifiedShuffleSplit over ntrials
    sss = StratifiedShuffleSplit(ntrials, test_size)
    train_errs = []
    test_errs = []
    f1_scores = []
    for train_index, test_index in sss.split(X, y):
      xtrain, xtest = X[train_index], X[test_index]
      ytrain, ytest = y[train_index], y[test_index]

      clf.fit(xtrain, ytrain)                
      y_pred = clf.predict(xtrain)    
      train_errs.append(1 - metrics.accuracy_score(ytrain, y_pred, normalize=True))
               
      y_pred = clf.predict(xtest)    
      test_errs.append(1 - metrics.accuracy_score(ytest, y_pred, normalize=True))  
      f1_scores.append(metrics.f1_score(ytest, y_pred, average='micro'))
    
    train_error = sum(train_errs) / len(train_errs)
    test_error = sum(test_errs) / len(test_errs)
    f1_score = sum(f1_scores) / len(f1_scores)

    return train_error, test_error, f1_score



def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()

def main():
    # load adult_subsample dataset with correct file path
    data_file =  "/content/drive/My Drive/ComSciM146/adult_subsample.csv"
    data = load_data(data_file, header=1, predict_col=-1)

    X = data.X; Xnames = data.Xnames
    y = data.y; yname = data.yname
    n,d = X.shape


    plt.figure()
    # plot histograms of each feature
    print('Plotting...')
    plot_histograms (X, y, Xnames=Xnames, yname=yname)
    

    # Preprocess X (e.g., normalize)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    

    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    # evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    # evaluate training error of Decision Tree classifier
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)




    # evaluate training error of k-Nearest Neighbors classifier
    print('Classifying using 3-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)

    print('Classifying using 5-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors = 5)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)

    print('Classifying using 7-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors = 7)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)



    # use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    mv = MajorityVoteClassifier()
    mv_train_err, mv_test_err, mv_f1_score = error(mv, X, y)
    print('\t-- Majority vote:')
    print('\t\t-- Training error: %.3f' % mv_train_err)
    print('\t\t-- Testing error: %.3f' % mv_test_err)
    print('\t\t-- F1_score: %.3f' % mv_f1_score)

    r = RandomClassifier()
    r_train_err, r_test_err, r_f1_score = error(r, X, y)
    print('\t-- Random classifier:')
    print('\t\t-- Training error: %.3f' % r_train_err)
    print('\t\t-- Testing error: %.3f' % r_test_err)
    print('\t\t-- F1_score: %.3f' % r_f1_score)

    dt = DecisionTreeClassifier(criterion = "entropy")
    dt_train_err, dt_test_err, dt_f1_score = error(dt, X, y)
    print('\t-- Decision tree:')
    print('\t\t-- Training error: %.3f' % dt_train_err)
    print('\t\t-- Testing error: %.3f' % dt_test_err)
    print('\t\t-- F1_score: %.3f' % dt_f1_score)
    
    kn = KNeighborsClassifier(n_neighbors = 5)
    kn_train_err, kn_test_err, kn_f1_score = error(kn, X, y)
    print('\t-- KNeighbors:')
    print('\t\t-- Training error: %.3f' % kn_train_err)
    print('\t\t-- Testing error: %.3f' % kn_test_err)
    print('\t\t-- F1_score: %.3f' % kn_f1_score)



    # use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k...')
    val_scores = []
    k_vals = []
    for i in range(25):
      k = 2*i + 1
      knc = KNeighborsClassifier(n_neighbors= k)
      scores = cross_val_score(knc, X, y, cv=10)
      val_scores.append(sum(scores) / len(scores))
      k_vals.append(k)

    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(k_vals, val_scores, color='r')
    ax.set_xlabel('K-Values')
    ax.set_ylabel('Validation Score')
    ax.set_title('K Value vs Validation Score')
    plt.show()


  
    # investigate decision tree classifier with various depths
    print('Investigating depths...')
    train_errors = []
    test_errors = []
    depth = []
    for i in range(1,21):
      dt = DecisionTreeClassifier(criterion="entropy", max_depth=i)
      train_error, test_error, f1 = error(dt, X, y)
      train_errors.append(train_error)
      test_errors.append(test_error)
      depth.append(i)
    
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(depth, test_errors, color='r', label='Test error')
    ax.scatter(depth, train_errors, color='b', label='Train error')
    ax.legend()
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Max Depth')
    plt.show()




    # investigate decision tree and k-Nearest Neighbors classifier with various training set sizes
    def learningCurveHelper(clf, x_train, y_train, x_test, y_test, n_trials = 10):
      training_err = []
      testing_err = []
      data_percentage = []
      for i in range(10):
        train_errs = []
        test_errs = []
        train_percent = 0.1 * i + 0.1
        for i in range(n_trials):
          if(train_percent != 1):
            for train_index, test_endex in StratifiedShuffleSplit(1, train_size = train_percent).split(x_train, y_train):
              x_train_subset, y_train_subset = x_train[train_index], y_train[train_index]
          else:
            x_train_subset = x_train
            y_train_subset = y_train

          clf.fit(x_train_subset, y_train_subset)

          y_train_pred = clf.predict(x_train_subset)
          train_errs.append(1 - metrics.accuracy_score(y_train_subset, y_train_pred, normalize=True))

          y_test_pred = clf.predict(x_test)
          test_errs.append(1 - metrics.accuracy_score(ytest, y_test_pred, normalize=True))  
        training_err.append(sum(train_errs) / len(train_errs))
        testing_err.append(sum(test_errs) / len(test_errs))
        data_percentage.append(train_percent)
      
      return training_err, testing_err, data_percentage
          
    #Split data into 90% train and 10% test       
    sss = StratifiedShuffleSplit(1, 0.1)   
    for train_index, test_index in sss.split(X, y):
      xtrain, xtest = X[train_index], X[test_index]
      ytrain, ytest = y[train_index], y[test_index]

    dTree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    t_train_error, t_test_error, t_percentages = learningCurveHelper(dTree, xtrain, ytrain, xtest, ytest)
    knn = KNeighborsClassifier(n_neighbors=15)
    k_train_error, k_test_error, k_percentages = learningCurveHelper(knn, xtrain, ytrain, xtest, ytest)

    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(t_percentages, t_train_error, color='r', label='Decision Tree Train Error')
    ax.scatter(t_percentages, t_test_error, color='b', label='Decision Tree Test Error')
    ax.scatter(t_percentages, k_train_error, color='k', label='KNN Train Error')
    ax.scatter(t_percentages, k_test_error, color='g', label='KNN Test Error')
    ax.legend()
    ax.set_xlabel('Percent of Training Data Used')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Amount of Training Data')
    plt.show()


    print('Done')


if __name__ == "__main__":
    main()