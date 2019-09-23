import os
import re
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#função de criação da nossa rede neural convolucional
def createCNN():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model

#função de CNN de artigo (não implementada)
# def createArtigoCNN():
#     model = Sequential()
#     model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#     model.add(Dense(128, activation=tf.nn.relu))
#     model.add(Dropout(0.2))
#     model.add(Dense(10,activation=tf.nn.softmax))
#     return model

#função de exibição de gráfico de accurácia por epochs
def printGraphicAcc(model_info):
    print(model_info.history.keys())
    plt.plot(model_info.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('matrics')
    plt.xlabel('epoch')
    plt.legend(['accuracy'], loc='upper left')
    plt.show()    #plt.legend(['accuracy', 'loss'], loc='upper left')

#função de exibição de gráfico de loss por epochs
def printGraphicLoss(model_info):
    print(model_info.history.keys())
    plt.plot(model_info.history['loss'])
    plt.title('model loss')
    plt.ylabel('matrics')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()


graphical=True

if __name__ == '__main__':
    #carregamento dos dados da base do mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #adaptação dos dados para entrada em funções do keras (os dados de entrada devem ter 4 dimensões) e ser do tipo float
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #normalização dos dados para entrada no nosso modelo
    #como se trata de imagens em tons de cinza, todas as variáveis vao de 0 ate 255
    x_train /= 255
    x_test /= 255


    epochs=[1,2,3,4,5]
    results_table=[]

    #laço principal para fazer a avaliação dos resultados variando a quantidade de epochs utilizadas na CNN
    for e in epochs:
        result=[e]
        model=createCNN()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model_info=model.fit(x=x_train,y=y_train, epochs=e)
        print(model.evaluate(x_test, y_test))

        #salva resultados dos modelos
        for i in model.evaluate(x_test, y_test):
            result.append(i)
        results_table.append(result)
        if graphical:
            printGraphicAcc(model_info)
            printGraphicLoss(model_info)

    #printa tabela organizada com métricas dos testes
    print("table of results")
    column=['epochs', 'loss', 'accuracy']
    df = pd.DataFrame(results_table, columns=column)
    print(df)
