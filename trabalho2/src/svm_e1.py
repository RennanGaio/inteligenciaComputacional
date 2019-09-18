import os
import re
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
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

graphical=True


def plotBeautifulBundry(X, y):
    print()

def plotsimplescatter(X, y):
    X.plot.scatter(x='At1', y ='At2', c=y, marker="x", colormap='viridis')
    plt.show()

def plotsuportvectors(suportIndices, dataX, datay):
    #pegar os indices, criar uma lista auxiliar, setar para 0 o decision bundry, jogar essa lista no c
    labelsWithBundry = np.copy(datay)
    for i in suportIndices:
        labelsWithBundry[i] = 0
    plotsimplescatter(dataX, labelsWithBundry)
    #df = pd.DataFrame(clf.best_estimator_.support_vectors_)
    #df.columns = ['a', 'b']
    #df.plot.scatter(x='a', y='b', c='black')

if __name__ == '__main__':
    if graphical:
        plotsimplescatter(data['X'], data['y'])



    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.5, 0.01]},
                        {'kernel': ['sigmoid'], 'gamma': [1, 0.5, 0.01]},
                        {'kernel': ['linear']},
                        {'kernel': ['poly'], 'degree': [3, 4, 5], 'gamma': [1, 0.5, 0.01]}
                       ]

    ks=[2,5,10]
    metrics = [0,""]

    classifiers=["SVM", "RF"]

    for classifier in classifiers:
        #this loop will interate for each classifier using diferents kfolds
        for kn in ks:
            print ("config: ",classifier)
            print ("using kfold= ",kf)
            print ("\n")
            #save time to metrics
            start = datetime.now()

            #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
            #SVM
            if classifier == "SVM":
              clf = GridSearchCV(estimator=SVC(C=1), param_grid=tuned_parameters, scoring="accuracy", n_jobs=-1, cv=kf, verbose=0)
            #XGBoost
            elif classifier == "RF":
              estimators = [ e for e in range(5, 25, 5) ]
              clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=dict(n_estimators=estimators), scoring="accuracy", n_jobs=-1, cv=kf, verbose=0)

            #this fit will train your classifier to your dataset
            clf.fit(data['X_train'], data['y_train'])
            end = datetime.now()

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("accuracy of %0.3f (+/-%0.03f), Ein of %0.3f for %r"
                      % (mean, std * 2, 1-mean, params))
            print()

            # table_of_contents = pd.DataFrame(clf.cv_results_)
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #     print(table_of_contents)
            #this will return the probability of each example be on all classes
            #y_pred = clf.predict_proba(x_train[val_index])
            y_pred = clf.best_estimator_.predict(data['X_test'])
            y_true = data['y_test']
            #this will generate the accuracy score

            accuracy = accuracy_score(y_true, y_pred)
            print ("accuracy: ", str(accuracy))
            print ("time consuption: " + str(end-start))
            print ("###########################\n")

            if graphical and classifier == "SVM":
                #pegar os indices, criar uma lista auxiliar, setar para 0 o decision bundry, jogar essa lista no c
                #Nesta primeira chamada ele irá criar o grafico com o melhor modelo obtido no gridSearch
                bestSuportIndices = clf.best_estimator_.support_
                plotsuportvectors(bestSuportIndices, data['X'], data['y'])

                #Nesta segunda chamada ele irá criar um gráfico com o pior modelo obtido pelo gridSearch


            #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file

            if metrics[0]<accuracy:
                metrics=[accuracy, clf, classifier, clf.best_estimator_]


    #save the best classificator into an file
    pkl_filename = "../models/pkl_bm_"+str(metrics[2])+"_"+str(datetime.now())+".pkl"
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
