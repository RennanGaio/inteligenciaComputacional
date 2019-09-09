import os
import re
import numpy as np
import pickle
from datetime import datetime
#import pandas as pd
#import matplotlib.pyplot as plt
#from PIL import Image
#import glob
#import cv2
#from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dropout, Dense
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential, load_model
# from keras.applications import VGG16
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
import random

from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb

from dataset import data


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.5, 0.01]},
                    {'kernel': ['sigmoid'], 'gamma': [1, 0.5, 0.01]},
                    {'kernel': ['linear']},
                    {'kernel': ['poly'], 'degree': [3, 4, 5]}
                   ]


ks=[2,5,10]
metrics = [0,""]

configuracoes=["SVN", "XGB"]

for config in configuracoes:
    #this loop will interate for each classifier using diferents kfolds
    for kn in ks:
        kf = KFold(kn, shuffle=True)
        for train_index, val_index in kf.split(x_train):
            print ("config: ",config)
            print ("using kfold= ",kn)
            print ("\n")
            #save time to metrics
            start = datetime.now()

            #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
            #SVN
            if config == "SVN":
              clf = GridSearchCV(estimator=SVC(C=1), tuned_parameters, scoring="accuracy", n_jobs=-1, cv=5, verbose=0)
            #XGBoost
            elif classifier == "XGBoost":
              clf = xgb.XGBClassifier(n_jobs=-1)
            # elif config == "polinomial":
            #     Cs = [ 2.0**c for c in range(-5, 15, 1) ]
            #     Gs = [ 2.0**g for g in range(3, -15, -2) ]
            #     kernels = [ 'rbf', 'poly', 'sigmoid' ]
            #     decision_function_shapes = [ 'ovo', 'ovr' ]
            #     clf = GridSearchCV(estimator=SVC(probability=False), param_grid=dict(kernel=kernels, C=Cs, gamma=Gs, decision_function_shape=decision_function_shapes), scoring="accuracy", n_jobs=-1, cv=5, verbose=0)

            #this fit will train your classifier to your dataset
            clf.fit(x_train[train_index],y_train[train_index])
            end = datetime.now()

            if config == "SVN":
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
            y_pred = clf.predict(x_train[val_index])

            #this will generate the accuracy score

            accuracy = accuracy_score(y_train[val_index], y_pred)
            print ("accuracy: ", str(accuracy))
            print ("time consuption: " + str(end-start))
            print ("###########################\n")


            #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file
            if (classifier!="XGBoost"):
                if metrics[0]<accuracy:
                    metrics=[accuracy, clf, clf.best_estimator_, x_train[train_index], y_train[train_index]]
            else:
                if metrics[0]<accuracy:
                    metrics=[accuracy, clf, "xgboost",x_train[train_index], y_train[train_index]]

#save the best classificator into an file
pkl_filename = "../models/pkl_best_model_"+str(metrics[2])+"date"+str(datetime.now())+".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(metrics[1], file)


#he must use the classifier with the best score
if metrics[1]=="xgboost":
    clf_greater=xgb.XGBClassifier()
else:
    clf_greater=metrics[1]
#clf_greater = metrics[1]
clf_greater.fit(metrics[3], metrics[4])

test_y_pred = clf_greater.predict(x_test)
test_accuracy = accuracy_score(y_test, test_y_pred)

#generete logs, with statistics
print ("best estimator: ", metrics[2])
print ("greater accuracy in validation: ", metrics[0])
print("accuracy in test:", test_accuracy)
file_name="../results/statistics-"+str(metrics[2])+".txt"
with open(file_name, "a+") as f:
    f.write("###################################\n")
    f.write("\nbest estimator: "+ str(metrics[1]))
    f.write("\ngreater accuracy in validation: "+ str(metrics[0]))
    f.write("\naccuracy in test:"+ str(test_accuracy))
    f.write("\nconjunto de teste: "+ str(metrics[3]))
    f.write("\nconjunto de treino: "+ str(metrics[4]))
    f.write("\n###################################\n")
