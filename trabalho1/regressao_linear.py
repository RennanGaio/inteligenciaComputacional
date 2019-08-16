# -*- coding: utf-8 -*-
"""
Aluno: Rennan de Lucena Gaio
Codigo referente aos exercicios 5,6 e 7 da lista 1 de IC2
"""

import numpy as np
import random
import matplotlib.pyplot as plt


'''funcoes auxiliáres para geracao de dados e funcao target (f)'''
def create_target():
    p1=[random.uniform(-1,1), random.uniform(-1,1)]
    p2=[random.uniform(-1,1), random.uniform(-1,1)]

    #target function = W1*x + W2*y + W0

    W1=p1[0]-p2[0]
    W2=p1[1]-p2[1]
    W0=p1[0]*p2[1]-p2[0]*p1[1]

    target_f=np.array([W0,W1,W2])

    return target_f

#função tem por objetivo transformar os pesos da funcao de uma maneira mais facil de representar passando por argumento no matplotlib
def transform_w_to_equation(w):
    #funçao de pesos w = W1*x + W2*y + W0
    #tranformar na equação y=Ax+b
    A=-w[1]/w[2]
    b=-w[0]/w[2]

    return A,b

#define as labels dos nossos exemplos com a funcao f que foi definida previamente de forma aleatoria
def pre_classify(target_f, point):
    if (np.inner(target_f, point) > 0):
        return 1
    else:
        return -1

#funcao que gera os dados de forma aleatoria para fazer os experimentos
def load_data(n, target_f):
    data=[]
    labels=[]
    for i in range(n):
        new_point=[1, random.uniform(-1,1), random.uniform(-1,1)]
        data.append(new_point)
        labels.append(pre_classify(target_f, new_point))
    return np.array(data), np.array(labels)


'''classe da regressao linear para aprendizado'''

class LinearRegression:
    """
    Implementação da regressao linear
    """
    def __init__(self, LR=0.1, threshold=10000):
        self.w = np.zeros(3)
        self.LR=LR
        self.threshold=threshold


    def predict(self, data):
        predicted_labels=[]
        for point in data:
            if (np.inner(self.w, point) > 0):
                predicted_labels.append(1)
            else:
                predicted_labels.append(-1)
        return predicted_labels


    def fit2d(self, X, y):
        n = np.size(X)

        m_x = np.mean(X[:,1])
        m_y = np.mean(X[:,2])

        SS_xy = np.sum(X[:,2]*X[:,1]) - n*m_y*m_x
        SS_xx = np.sum(X[:,1]*X[:,1]) - n*m_y*m_x

        self.w[0] = SS_xy/SS_xx
        self.w[1] = m_y - self.w[0]*m_x
        return self.w

    def fit(self, X, y):
        A=(X.T)@X
        B=(X.T)@y
        self.w= np.linalg.solve(A, B)
        return self.w


'''classe do Perceptron para aprendizado'''

class Perceptron:
    """
    Implementação do perceptron
    """
    def __init__(self, w , LR=0.1, threshold=10000):
        self.w = w
        self.LR=LR
        self.threshold=threshold


    def predict(self, data):
        predicted_labels=[]
        for point in data:
            if (np.inner(self.w, point) > 0):
                predicted_labels.append(1)
            else:
                predicted_labels.append(-1)
        return predicted_labels


    def fit(self, X, y):
        interactions=0
        while interactions<self.threshold:
            interactions+=1
            saved_w=np.copy(self.w)
            random_index_order=random.sample(range(len(y)), len(y))
            for index in random_index_order:
                prediction=self.predict([X[index]])[0]
                if prediction!= y[index]:
                    self.w+=self.LR*(y[index]-prediction)*X[index]
                    break
            if np.array_equal(saved_w,self.w):
                break
        return self.w, interactions


def exercise_5(N=100):
    Ein=[]
    for j in range(1000):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        classifier=LinearRegression()
        g_function = classifier.fit(data, labels)

        predicted_labels=classifier.predict(data)

        E=0
        for label, p_label in zip(labels, predicted_labels):
            if label!=p_label:
                E+=1./N

        Ein.append(E)

    Ein_mean=np.mean(np.array(Ein))
    print("Ein media")
    print(Ein_mean)

def exercise_6(N=1000):
    Eout=[]
    for j in range(1000):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        classifier=LinearRegression()
        g_function = classifier.fit(data, labels)

        #cria mais amostras fora das amostras uzadas para o treino (out of sample)
        data, labels = load_data(N, target_f)

        predicted_labels=classifier.predict(data)

        E=0
        for label, p_label in zip(labels, predicted_labels):
            if label!=p_label:
                E+=1./N
        if (j%100 == 0):
            print("rodada: ",j)

        Eout.append(E)
    Eout_mean=np.mean(np.array(Eout))

    print("Eout medio")
    print(Eout_mean)

def exercise_7(N=10):
    interactions=[]
    for j in range(1000):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        classifier=LinearRegression()
        g_function = classifier.fit(data, labels)

        classifier=Perceptron(w=g_function)
        g_function, i = classifier.fit(data, labels)

        interactions.append(i)

        if (j%100 == 0):
            print("rodada: ",j)

    interactions_mean=np.mean(np.array(interactions))

    print("media de iteracoes")
    print(interactions_mean)


if __name__ == '__main__':
    graphical=False

    if graphical:
        N=10
        """### Visualização dos nossos dados"""
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #funcao que ira mostrar nossa linha da target function
        x=np.linspace(-1,1,100)
        A,b = transform_w_to_equation(target_f)
        plt.plot(x, A*x+b, '-r', label='target_function')

        #nossos dados rotulados
        plt.scatter(data[:,1], data[:,2], c=labels, cmap=plt.cm.Spectral)

        classifier=LinearRegression()
        g_function = classifier.fit(data, labels)

        #funcao que ira mostrar nossa linha da g function
        x=np.linspace(-1,1,100)
        A,b = transform_w_to_equation(g_function)
        #A,b = transform_w_to_equation(g_function)
        plt.plot(x, A*x+b, '-g', label='g_function')

        plt.show()
    else:
        exercise_5()
        #resposta C
        exercise_6()
        #resposta C
        exercise_7()
        #resposta A
