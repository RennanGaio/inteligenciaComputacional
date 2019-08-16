# -*- coding: utf-8 -*-
"""
Aluno: Rennan de Lucena Gaio
Codigo referente aos exercicios 5,6 e 7 da lista 1 de IC2
"""

import numpy as np
import random
import matplotlib.pyplot as plt
N=1000

'''funcoes auxiliáres para geracao de dados e funcao target (f)'''
def create_target():
    #target function = W1*x^2 + W2*y^2 + W0
    #W1 = 1, W2 = 1 e W0 = -0.6
    W1=1
    W2=1
    W0=-0.6

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
def pre_classify(point):
    if ((point[1]**2 + point[2]**2 - 0.6) > 0):
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
        labels.append(pre_classify(new_point))
    return np.array(data), np.array(labels)


'''classe da regressao linear para aprendizado'''

class LinearRegression:
    """
    Implementação da regressao linear
    """
    def __init__(self, LR=0.1, threshold=10000):
        self.w = np.zeros(2)
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

def exercise_8():
    Ein=[]
    test_rounds=100
    for j in range(test_rounds):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #geração de ruido da funcao
        ten_percent=int(N/10)
        for i in range(ten_percent):
            random_index=random.randrange(len(labels))
            labels[random_index]=random.sample([-1,1],1)[0]


        classifier=LinearRegression()
        g_function = classifier.fit(data, labels)
        predicted_labels = classifier.predict(data)

        #calcura o erro dentro da amostra para cada teste
        E=0.
        for label, predicted_label in zip(labels, predicted_labels):
            if label!=predicted_label:
                E+=1./N
        Ein.append(E)

    Ein_mean=np.mean(np.array(Ein))

    print("media de erro dentro da amostra")
    print(Ein_mean)

    ##resposta D

def exercise_9():
    target_f=create_target()
    data, labels = load_data(N, target_f)

    #estaremos aqui ampliando o vetor de dados para conseguirmos suportar as variaveis de ordem maior
    #nosso vetor de data vai passar de (1,x1,x2) para (1,x1,x2,x1x2,x1^2, x2^2)
    new_data =[]
    for point in data:
        new_features=np.array([point[1]*point[2], point[1]**2, point[2]**2])
        new_data.append(np.append(point, new_features))

    #geração de ruido da funcao
    ten_percent=int(N/10)
    for i in range(ten_percent):
        random_index=random.randrange(len(labels))
        labels[random_index]=random.sample([-1,1],1)[0]


    classifier=LinearRegression()
    g_function = classifier.fit(np.array(new_data), labels)
    print(g_function)
    ##resposta letra A


def exercise_10():
    target_f=create_target()
    data, labels = load_data(N, target_f)

    #estaremos aqui ampliando o vetor de dados para conseguirmos suportar as variaveis de ordem maior
    #nosso vetor de data vai passar de (1,x1,x2) para (1,x1,x2,x1x2,x1^2, x2^2)
    new_data =[]
    for point in data:
        new_features=np.array([point[1]*point[2], point[1]**2, point[2]**2])
        new_data.append(np.append(point, new_features))

    #geração de ruido da funcao
    ten_percent=int(N/10)
    for i in range(ten_percent):
        random_index=random.randrange(len(labels))
        labels[random_index]=random.sample([-1,1],1)[0]


    classifier=LinearRegression()
    g_function = classifier.fit(np.array(new_data), labels)

    test_rounds=1000
    for i in range(test_rounds):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #estaremos aqui ampliando o vetor de dados para conseguirmos suportar as variaveis de ordem maior
        #nosso vetor de data vai passar de (1,x1,x2) para (1,x1,x2,x1x2,x1^2, x2^2)
        new_data =[]
        for point in data:
            new_features=np.array([point[1]*point[2], point[1]**2, point[2]**2])
            new_data.append(np.append(point, new_features))

        #geração de ruido da funcao
        ten_percent=int(N/10)
        for i in range(ten_percent):
            random_index=random.randrange(len(labels))
            labels[random_index]=random.sample([-1,1],1)[0]

        predicted_labels=classifier.predict(np.array(new_data))

        Eout=[]
        E=0.
        for label, predicted_label in zip(labels, predicted_labels):
            if label!=predicted_label:
                E+=1./N
        Eout.append(E)

    Eout_mean=np.mean(np.array(Eout))

    print("media de erro fora da amostra")
    print(Eout_mean)
    ###resposta B




if __name__ == '__main__':

    graphical=False

    if graphical:
        """### Visualização dos nossos dados"""
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #geração de ruido da funcao
        ten_percent=int(N/10)
        for i in range(ten_percent):
            random_index=random.randrange(len(labels))
            labels[random_index]=random.sample([-1,1],1)

        #funcao que ira mostrar nossa linha da target function
        x=np.linspace(-1,1,100)
        y=np.linspace(-1,1,100)
        X, Y = np.meshgrid(x, y)
        F=np.sign(X*X + Y*Y - 0.6)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(X, Y, F)
        plt.scatter(X,Y,F)
        plt.show()

    else:
        exercise_8()
        exercise_9()
        exercise_10()
