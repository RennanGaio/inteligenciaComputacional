import os
import re
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
#import matplotlib.pyplot as plt
#from PIL import Image
#import glob
#import cv2
#from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

#from sklearn import preprocessing
#from sklearn import neighbors, datasets
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn import decomposition
import random

from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb

from dataset import data


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.5, 0.01]},
                    {'kernel': ['sigmoid'], 'gamma': [1, 0.5, 0.01]},
                    {'kernel': ['linear']},
                    {'kernel': ['poly'], 'degree': [3, 4, 5], 'gamma': [1, 0.5, 0.01]}
                   ]

ks=[2,5,10]
metrics = [0,""]

classifiers=["SVN", "XGB"]

for classifier in classifiers:
    #this loop will interate for each classifier using diferents kfolds
    for kf in ks:
        print ("config: ",classifier)
        print ("using kfold= ",kf)
        print ("\n")
        #save time to metrics
        start = datetime.now()

        #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
        #SVN
        if classifier == "SVN":
          clf = GridSearchCV(estimator=SVC(C=1), param_grid=tuned_parameters, scoring="accuracy", n_jobs=-1, cv=kf, verbose=0)
        #XGBoost
        elif classifier == "XGBoost":
          clf = xgb.XGBClassifier(n_jobs=-1)

        #this fit will train your classifier to your dataset
        clf.fit(data['X_train'], data['y_train'])
        end = datetime.now()

        if classifier == "SVN":
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        #this will return the probability of each example be on all classes
        #y_pred = clf.predict_proba(x_train[val_index])
        y_pred = clf.predict(data['X_test'])
        y_true = data['y_test']
        #this will generate the accuracy score

        accuracy = accuracy_score(y_true, y_pred)
        print ("accuracy: ", str(accuracy))
        print ("time consuption: " + str(end-start))
        print ("###########################\n")


        #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file

        if metrics[0]<accuracy:
            metrics=[accuracy, clf, classifier, clf.best_estimator_]


#save the best classificator into an file
pkl_filename = "../models/pkl_bm_"+str(metrics[2])+"date"+str(datetime.now())+".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(metrics[1], file)


# #he must use the classifier with the best score
# clf_greater=metrics[1]
#
#
# test_y_pred = clf_greater.predict(x_test)
# test_accuracy = accuracy_score(y_test, test_y_pred)

#generete logs, with statistics
print ("best estimator: ", metrics[2])
print ("greater accuracy in validation: ", metrics[0])
#print("accuracy in test:", test_accuracy)
file_name="../results/statistics-"+str(metrics[2])+".txt"
with open(file_name, "a+") as f:
    f.write("###################################\n")
    f.write("\nbest estimator: "+ str(metrics[1]))
    #f.write("\ngreater accuracy in validation: "+ str(metrics[0]))
    f.write("\naccuracy in test:"+ str(metrics[0]))
    f.write("\n###################################\n")
