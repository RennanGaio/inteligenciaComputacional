import os
import re
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
import random

from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb

#função para carregar os datasets nas variáveis
def load_dataset(dataset):
    data_train = pd.read_csv("../data/"+dataset+"data_train.csv", sep='\t', index_col=0)
    #print(data_train.shape)
    #print(data_train.columns)
    X_train = data_train.drop('Class', axis=1)
    y_train = data_train['Class']

    data_test = pd.read_csv("../data/"+dataset+"data_test.csv", sep='\t', index_col=0)
    #print(data_test.shape)
    X_test = data_test.drop('Class', axis=1)
    y_test = data_test['Class']

    dataset_data = pd.read_csv("../data/"+dataset+".csv")
    X = dataset_data.drop('Class', axis=1)
    y = dataset_data['Class']

    return X_train, y_train, X_test, y_test, X, y

#Inicialização das listas de suporte
datasets=["phoneme","wine","heart","coil2000","magic"]
svm_param={'kernel': 'rbf', 'gamma': 1}
results_table=[]

if __name__ == '__main__':
    #classifierxgb = pickle.load(open("../models/XGBoost_2019-09-19 11:27:24.336451.pkl", "rb"))
    #classifiersvm = pickle.load(open("../models/pkl_bm_SVM_2019-09-19 11:27:24.336147.pkl", "rb"))

    #Para cada dataset, será utilizado os classificadores com os melhores hiperparametros encontrados no Experimento 1 do trabalho
    for dataset in datasets:
        X_train, y_train, X_test, y_test, X, y = load_dataset(dataset)

        classifierxgb = xgb.XGBClassifier(n_jobs=-1)
        classifiersvm = SVC(**svm_param)
        #this fit will train your classifier to your dataset
        classifierxgb.fit(X_train, y_train)
        classifiersvm.fit(X_train, y_train)

        y_pred_xgb = classifierxgb.predict(X_test)
        y_pred_svm = classifiersvm.predict(X_test)

        #this will generate the accuracy score out of sample

        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)


        result=['XGBoost', dataset, accuracy_xgb, 1-accuracy_xgb]
        results_table.append(result)
        result=['SVM', dataset, accuracy_svm, 1-accuracy_svm]
        results_table.append(result)

    #exibe resultados organizados de cada classificador para cada dataset
    print("table of results")
    column=['model', 'dataset', 'accuracy', 'Eout']
    df = pd.DataFrame(results_table, columns=column)
    print(df)
