# -*- coding: utf-8 -*-

'''Aluno: Rennan de Lucena Gaio '''

import numpy as np
import random
import matplotlib.pyplot as plt
N=10

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


'''Para termos um código semelhante com os utilizados no sklearn modelei o classificador como uma classe, dessa forma
teremos um "cluster" que irá guardar o classificador do treino, e poderá ser utilizado para qualquer outro dado de testeself.

Para que isso funcione, as 2 principais funções dentro do classificador serão: fit e predict.
Um usuário final para esse código só irá se preocupar em passar os dados de treino para a função fit
e depois passar para a função predict os dados de teste, tendo como retorno a probabilidade do determinado dado estar na classe.
'''

class LogisticRegression:
    #inicialização da classe, setando parametros como taxa de aprendizado e bias dos dados
    def __init__(self, learning_rate=0.01, num_iter=100000, bias=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.bias = bias
        self.verbose = verbose

    def add_bias(self, X):
        #adiciona uma coluna de 1s aos dados de X para servirem como bias
        bias = np.ones((X.shape[0], 1))

        #if self.verbose:
        #    print "DADOS DE ENTRADA"
        #    print np.concatenate((bias, X), axis=1)

        return np.concatenate((bias, X), axis=1)

    #função sigmoide phi
    def phi(self, z):
        return 1 / (1 + np.exp(-z))

    #função erro utilizada correlacionada com a funcao gradiente()
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    ########## teste de funcao gradiente #############

    #essa implementação utiliza um parâmetro lambda de regularização, porém eu nao entendi exatamente para o que serve
    #Como eu não estava certo de sua funcionalidade eu a descartei até segunda ordem
    def gradiente1(self, X, y, z,lam=0.1):
        #temporary weight vector
        w1 = copy.copy(self.theta) #import copy to create a true copy
        w1[0] = 0
        #calc gradient
        grad = (np.dot((phi(z) - y).T,X).T) + lam * w1
        return grad

    #Essa e a funcao erro utilizada ate o momento. Sua derivada deveria ter um fator 1/m na frente, mas como estamos utilizando learning rate
    #não botarei esse fator, e irei controlar o tamanho do vetor apartir dessa variavel criada
    def gradiente(self, X, y, z):
        return np.dot(X.T, (self.phi(z) - y)) / y.size

    #utilizara o gradiente descendente para encontrar o erro minimo da funcao phi
    #inicializando o vetor de pesos zerado, ele irá se ajustar aos dados até que ele consiga separar de forma adequada os dados
    #utilizando um determinado numero de passos
    def fit(self, X, y):
        if self.bias:
            X = self.add_bias(X)

        # inicialização do vetor de pesos
        self.theta = np.zeros(X.shape[1])

        #erro anterior para comparar a evolucao do erro
        loss_anterior=0
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)

            #calcula a funcao gradiente baseada na derivada da funcao de perda (loss)
            gradient = self.gradiente(X,y,z)

            #anda com o vetor de pesos na direçao contraria do gradiente (pois queremos minimizar nossa funcao, nao maximizar)
            self.theta -= self.learning_rate * gradient

            #debug para saber o quanto o erro está se aproximando, nessa etapa, se a diferença do erro ja estivesse pequena
            #eu poderia travar o laço do for, e fazer com que meu algoritmo rodasse com menos iterações sem afetar muito seu desempenho

            #porém de acordo com alguns testes feitos eu vi que ficar calculando o erro em cada iteração deixa o algoritmo bem lento,
            #logo esse recurso seria interessante apenas se o erro convergisse rápido (o q paresce ser verdade)
            if(self.verbose == True and i % 100 == 0):
                z = np.dot(X, self.theta)
                print "loss: ", str(self.loss(self.phi(z), y))
                print "diferença de erro:", str(loss_anterior-self.loss(self.phi(z), y))
                loss_anterior=self.loss(self.phi(z), y)

    #aplica os dados de teste na funcao phi com os pesos ajustados e retorna a probabilidade dos dados estarem em cada classe
    #quanto mais perto de 1, mais ele pertence a classe 1, quanto mais perto de 0, mas ele pertence a classe 0
    def predict_proba(self, X):
        if self.bias:
            X = self.add_bias(X)

        return self.phi(np.dot(X, self.theta))


if __name__ == '__main__':
    """### Visualização dos nossos dados"""
    target_f=create_target()
    data, labels = load_data(N, target_f)

    rodada=0

    #para cada embaralhamento do kfold ele irá medir qual conjunto de dados obteve o melhor AUC, e ficar com esse para ser utilizado futuramente
    for train_index, test_index in kf.split(X):
        rodada+=1

        #carrega a classe do classificador com os dados de treino e testa a cada iteração para conseguir ajustar ao melhor conjunto
        clf=LogisticRegression(verbose=False)
        clf.fit(X[train_index],y[train_index])
        y_pred = clf.predict_proba(X[test_index])

        print y_pred
        print y[test_index]

        #calcula o AUC do conjunto dado
        auc = roc_auc_score(y[test_index], y_pred)
        print "AUC: ", str(auc)
        print "###########################\n"

        #atualiza o melhor AUC caso necessário
        if best_auc[0] < auc:
            best_auc=[auc, rodada]

    print "melhor AUC: ", best_auc[0]
    print "rodada: ", best_auc[1]
