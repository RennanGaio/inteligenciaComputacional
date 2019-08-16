# -*- coding: utf-8 -*-

'''
Aluno: Rennan de Lucena Gaio
Codigo referente aos 4 exercícios 13 e 14 da lista 1 de IC2
'''

import numpy as np
import random
import matplotlib.pyplot as plt
N=100

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
    def __init__(self, learning_rate=0.01, num_iter=100000, verbose=False, lenght=3):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.verbose = verbose
        self.w = np.zeros(lenght)


    #função sigmoide phi
    def phi(self, z):
        return 1 / (1 + np.exp(-z))

    #função erro utilizada correlacionada com a funcao gradiente()
    # def loss(self, h, y):
    #     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def loss(self,x, y):
        return np.log(1.0 + np.exp(-y * np.dot(self.w,x)))

    ########## teste de funcao gradiente #############

    #Essa e a funcao erro utilizada ate o momento. Sua derivada deveria ter um fator 1/m na frente, mas como estamos utilizando learning rate
    #não botarei esse fator, e irei controlar o tamanho do vetor apartir dessa variavel criada
    # def gradiente(self, X, y, z):
    #     return np.dot(X.T, (self.phi(z) - y)) / y.size

    def gradiente(self, x, y):
        return (-(y*x)/(1.0 + np.exp(y*np.dot(self.w,x))))

    #utilizara o gradiente descendente para encontrar o erro minimo da funcao phi
    #inicializando o vetor de pesos zerado, ele irá se ajustar aos dados até que ele consiga separar de forma adequada os dados
    #utilizando um determinado numero de passos
    def fit(self, X, y):
        # inicialização do vetor de pesos
        self.w = np.zeros(X.shape[1])
        interations=0

        for i in range(self.num_iter):
            prev_w=np.copy(self.w)
            random_index_order=random.sample(range(len(y)), len(y))
            interations+=1

            for index in random_index_order:
                #z = np.dot(X, self.w)

                #calcula a funcao gradiente baseada na derivada da funcao de perda (loss)
                gradient = self.gradiente(X[index],y[index])

                #anda com o vetor de pesos na direçao contraria do gradiente (pois queremos minimizar nossa funcao, nao maximizar)
                self.w -= self.learning_rate * gradient

            if( np.sqrt( np.sum((prev_w - self.w)**2) ) < 0.01):
                #print("atingiu o erro minimo")
                break


        return self.w, interations

    #aplica os dados de teste na funcao phi com os pesos ajustados e retorna a probabilidade dos dados estarem em cada classe
    #quanto mais perto de 1, mais ele pertence a classe 1, quanto mais perto de 0, mas ele pertence a classe 0
    def predict_proba(self, X):
        proba=self.phi(np.dot(X, self.w))
        return proba

    def predict(self, X):
        proba=self.phi(np.dot(X, self.w))
        predicted_labels=[]
        for prob in proba:
            if prob > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(-1)
        return predicted_labels

    def loss_mean(self, X, y):
        losses=[]
        for point, label in zip(X,y):
            losses.append(self.loss(point, label))
        return np.mean(np.array(losses))

if __name__ == '__main__':
    rounds=100
    Eout=[]
    interactions=[]
    #para cada embaralhamento do kfold ele irá medir qual conjunto de dados obteve o melhor AUC, e ficar com esse para ser utilizado futuramente
    for i in range(rounds):
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #carrega a classe do classificador com os dados de treino e testa a cada iteração para conseguir ajustar ao melhor conjunto
        classifier=LogisticRegression(verbose=False)
        g_function, i = classifier.fit(data,labels)
        interactions.append(i)

        #criação de novos dados para se fazer a medicao de divergencia entre as funcoes
        data, labels = load_data(N, target_f)
        Eout.append(classifier.loss_mean(data, labels))
        # predicted_labels = classifier.predict(data)
        #
        # #y_pred = clf.predict_proba(data)
        #
        # #variavel que guarda a divergencia a cada iteracao da comparacao
        # E=0.
        # for label, predicted_label in zip(labels, predicted_labels):
        #     if label!=predicted_label:
        #         E+=1./N
        # Eout.append(E)

    interactions_mean=np.mean(np.array(interactions))
    Eout_mean=np.mean(np.array(Eout))

    print("esperimento com N =", N)
    print("media de iteracoes")
    print(interactions_mean)
    print("Eout medio")
    print(Eout_mean)

    #respostas na lista: D e A respectivamente
