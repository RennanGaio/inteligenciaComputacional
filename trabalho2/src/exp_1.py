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

from dataset import data

#variáveis globias
graphical=True
verbose=False


#função não foi implementada ainda (seria um extra), lembrar para o futuro
# def plotBeautifulBundry(X, y):
#     print()

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


    #Inicialização de listas, variáveis e configurações que serão utilizadas
    tuned_parameters = [{'kernel': 'rbf', 'gamma': 1},
                        {'kernel': 'rbf', 'gamma': 'auto'},
                        {'kernel': 'sigmoid', 'gamma': 1},
                        {'kernel': 'sigmoid', 'gamma': 0.5},
                        {'kernel': 'sigmoid', 'gamma': 0.01},
                        {'kernel': 'linear', 'gamma':'auto'},
                        {'kernel': 'poly', 'degree': 3, 'gamma':'auto'},
                        {'kernel': 'poly', 'degree': 4, 'gamma':'auto'},
                        {'kernel': 'poly', 'degree': 5, 'gamma':'auto'}
                       ]

    ks=[2,5,10]
    best_metrics = [0,""]
    worse_metrics = [1,""]

    X_train=data['X_train'].to_numpy()
    y_train=data['y_train'].to_numpy()
    X_test=data['X_test'].to_numpy()
    y_test=data['y_test'].to_numpy()

    results_table=[]

    #Laço principal do programa, para cada configuração diferente de SVM ele irá treinar e guardar as metricas
    for parameter in tuned_parameters:
        #Loop de controle dos kfolds para fazer a crossvalidação
        for kn in ks:
            kf_accuracys=[]
            kf = KFold(kn, shuffle=True)
            print ("using SVM")
            print(parameter)
            print ("using kfold= ",kn)
            print ("\n")
            for train_index, val_index in kf.split(X_train):
                #salva métrica de tempo (acabou não sendo utilizada)
                start = datetime.now()

                #cria o classificador utilizando um dos parâmetros de teste
                clf = SVC(**parameter)
                #ajusta o modelo aos dados de treino
                clf.fit(X_train[train_index], y_train[train_index])

                end = datetime.now()

                #faz a predição dos dados de validação para avaliar a acuracia dentro da amostra
                y_pred_val = clf.predict(X_train[val_index])

                accuracy = accuracy_score(y_train[val_index], y_pred_val)
                if verbose:
                    print ("accuracy in sample: ", str(accuracy))
                    print ("time consuption: " + str(end-start))
                    print ("###########################\n")

                #faz a predição dos dados de teste para avaliar a acuracia fora da amostra
                y_pred_test = clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred_test)
                kf_accuracys.append(accuracy)
                if verbose:
                    print("OUT OF SAMPLE RESULTS")
                    print ("accuracy out of sample: ", str(accuracy))
                    print ("Eout: ", str(1-accuracy))
                    print ("###########################\n")

                #salva a melhor e pior configuração obtida pelo SVM para plotar gráficos
                if best_metrics[0]<accuracy:
                    best_metrics=[accuracy, clf, "SVM", parameter]

                if worse_metrics[0]>accuracy:
                    worse_metrics=[accuracy, clf, "SVM", parameter]

            #Salva e printa todas as métricas referentes aos kfold, com média e variância dos dados
            accuracy_array=np.array(kf_accuracys)
            print("BATCH FINISHED, RESULTS SUMARY")
            print("KF METRICS")
            print("Using kfold = ", kn)
            print("configuration:")
            print(parameter)
            print("Accuracy out of sample: %0.2f (+/- %0.2f)" % (accuracy_array.mean(), accuracy_array.std() * 2))
            print ("Eout: ", str(1-accuracy_array.mean()))
            print ("###########################\n")
            result=['SVM', parameter, kn, "%0.2f (+/- %0.2f)" % (accuracy_array.mean(), accuracy_array.std() * 2), 1-accuracy_array.mean()]
            results_table.append(result)


    #Exibe os gráficos de vetores de suporte
    if graphical:
        #pegar os indices, criar uma lista auxiliar, setar para 0 o decision bundry, jogar essa lista no c
        #Nesta primeira chamada ele irá criar o grafico com o melhor modelo obtido
        bestSuportIndices = best_metrics[1].support_
        if verbose:
            print("quantidade de vetores de suporte")
            print(len(bestSuportIndices))
        plotsuportvectors(bestSuportIndices, data['X'], data['y'])

        #Nesta segunda chamada ele irá criar um gráfico com o pior modelo obtido
        worseSuportIndices = worse_metrics[1].support_
        if verbose:
            print("quantidade de vetores de suporte")
            print(len(worseSuportIndices))
        plotsuportvectors(worseSuportIndices, data['X'], data['y'])

    #Experimento utilizando XGBoost para comparação
    #basicamente o procedimento é semelhante ao anterior pelo SVM, porém desta vez o classificador é o XGBoost
    kf = KFold(10, shuffle=True)
    print ("using XGBoost")
    print ("using kfold= 10")
    print ("\n")
    for train_index, val_index in kf.split(X_train):
        clf = xgb.XGBClassifier(n_jobs=-1)
        clf.fit(X_train[train_index], y_train[train_index])

        y_pred_val = clf.predict(X_train[val_index])

        accuracy = accuracy_score(y_train[val_index], y_pred_val)
        if verbose:
            print ("accuracy in sample: ", str(accuracy))
            print ("time consuption: " + str(end-start))
            print ("###########################\n")

        y_pred_test = clf.predict(X_test)
        #this will generate the accuracy score out of sample

        accuracy = accuracy_score(y_test, y_pred_test)
        kf_accuracys.append(accuracy)
        if verbose:
            print("OUT OF SAMPLE RESULTS")
            print ("accuracy out of sample: ", str(accuracy))
            print ("Eout: ", str(1-accuracy))
            print ("###########################\n")

    accuracy_array=np.array(kf_accuracys)
    print("BATCH FINISHED, RESULTS SUMARY")
    print("Accuracy out of sample: %0.2f (+/- %0.2f)" % (accuracy_array.mean(), accuracy_array.std() * 2))
    print ("Eout: ", str(1-accuracy_array.mean()))
    print ("###########################\n")
    result=['XGBoost', 'default', 10, "%0.2f (+/- %0.2f)" % (accuracy_array.mean(), accuracy_array.std() * 2), 1-accuracy_array.mean()]
    results_table.append(result)

    #salva o melhor classificador de SVM
    pkl_filename = "../models/pkl_bm_"+str(best_metrics[2])+"_"+str(datetime.now())+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(best_metrics[1], file)

    #salva o melhor classificador do XGBoost
    pkl_filename= "../models/XGBoost_"+str(datetime.now())+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    #exibição de estatisticas gerais
    print ("best SVM configuration: ", best_metrics[3])
    print ("greater SVM accuracy in test: ", best_metrics[0])


    #Exibição de tabela organizada com todos os experimentos executados
    print("table of results")
    column=['model', 'configuration', 'kfold', 'accuracy', 'Eout']
    df = pd.DataFrame(results_table, columns=column)
    print(df)

    #salva logs em arquivos caso necessário
    file_name="../results/statistics-"+str(best_metrics[2])+".txt"
    with open(file_name, "a+") as f:
        f.write("###################################\n")
        f.write("\nbest estimator: "+ str(best_metrics[1]))
        #f.write("\ngreater accuracy in validation: "+ str(metrics[0]))
        f.write("\naccuracy in test:"+ str(best_metrics[0]))
        f.write("\n###################################\n")
