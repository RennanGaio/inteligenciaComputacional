# -*- coding: utf-8 -*-

'''
Aluno: Rennan de Lucena Gaio
Codigo referente aos 4 exercícios 13 e 14 da lista 1 de IC2
'''

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

'''classe da regressao logistica para aprendizado'''

class LogisticRegression:
    #inicialização da classe, setando parametros como taxa de aprendizado, quantidade máxima de iterações e inicialização dos pesos w
    def __init__(self, learning_rate=0.01, num_iter=100000, verbose=False, lenght=3):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.verbose = verbose
        self.w = np.zeros(lenght)

    #função de perda Ein ou Eout para cada ponto da nossa função
    def loss(self,x, y):
        return np.log(1.0 + np.exp(-y * np.dot(self.w,x)))

    #gradiente da nossa função de perda para a minimização do erro
    def gradiente(self, x, y):
        return (-(y*x)/(1.0 + np.exp(y*np.dot(self.w,x))))

    #utilizara o gradiente descendente para encontrar o erro minimo da funcao de perda
    #inicializando o vetor de pesos zerado, ele irá se ajustar aos dados até que ele consiga separar de forma adequada os dados
    #utilizando um determinado numero de passos até que a norma da diferença dos pesos atual e anterior seja menor que 0.01
    def fit(self, X, y):
        # inicialização do vetor de pesos
        self.w = np.zeros(X.shape[1])
        interations=0

        for i in range(self.num_iter):
            #é utilizada a função de copy pois se não é criado um ponteiro para os pesos 2, e isso impediria que eu salvasse os dados da rodada passada
            prev_w=np.copy(self.w)
            #seleciona de forma aleatoria a ordem dos pontos que vai ser avaliado
            random_index_order=random.sample(range(len(y)), len(y))
            interations+=1
            for index in random_index_order:
                #calcula a funcao gradiente baseada na derivada da funcao de perda (loss)
                gradient = self.gradiente(X[index],y[index])

                #anda com o vetor de pesos na direçao contraria do gradiente (pois queremos minimizar nossa funcao, nao maximizar)
                self.w -= self.learning_rate * gradient

            #checa se a norma da diferença dos pesos atual e anterior seja menor que 0.01, caso seja, eu saio do laço principal
            if( np.sqrt( np.sum((prev_w - self.w)**2) ) < 0.01):
                #print("atingiu o erro minimo")
                break

        return self.w, interations

    #calcula o erro dentro ou fora da amostra para todos os pontos do dataset
    def loss_mean(self, X, y):
        losses=[]
        for point, label in zip(X,y):
            losses.append(self.loss(point, label))
        return np.mean(np.array(losses))

if __name__ == '__main__':
    #inicialização das variaveis e dos acumuladores de erro fora da amostra e das iterações por rodada de teste
    N=100
    rounds=100
    Eout=[]
    interactions=[]
    #laço das rodadas de teste
    for i in range(rounds):
        #inicialização dos dados
        target_f=create_target()
        data, labels = load_data(N, target_f)

        #carrega a classe do classificador com os dados de treino e testa a cada iteração para conseguir ajustar ao melhor conjunto
        classifier=LogisticRegression(verbose=False)
        g_function, i = classifier.fit(data,labels)
        #salva a quantidade de iterações para o algoritmo convergit
        interactions.append(i)

        #criação de novos dados para se fazer a medicao de divergencia entre as funcoes
        data, labels = load_data(N, target_f)
        #calcula o erro fora da amostra passando os dados gerados para a função de perda da classe
        Eout.append(classifier.loss_mean(data, labels))

    #faz a media dos valores obtidos por rodada
    interactions_mean=np.mean(np.array(interactions))
    Eout_mean=np.mean(np.array(Eout))

    print("esperimento com N =", N)
    print("exercicio 13")
    print("Eout medio")
    print(Eout_mean)
    print("exercicio 14")
    print("media de iteracoes")
    print(interactions_mean)


    #respostas na lista: D e A respectivamente
