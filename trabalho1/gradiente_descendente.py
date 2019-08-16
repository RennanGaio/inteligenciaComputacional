# -*- coding: utf-8 -*-
"""
Aluno: Rennan de Lucena Gaio
Codigo referente aos 4 exercícios 11 e 12 da lista 1 de IC2
"""

import numpy as np
import random
import matplotlib.pyplot as plt


'''
funções referentes ao gradiente descendente
não foi criada uma classe para esse problema por ele ser mais simples e as perguntas serem mais diretas
'''

#definição da nossa função de erro dada pelo enunciado
def error_function(u,v):
    return ( u*np.e**(v) - 2*v*np.e**(-u) )**2

#gradiente em relação a variavel u da nossa função de erro
def gradiente_u(u,v):
    return ( 2 * ((u*np.e**(v)) - (2*v*np.e**(-u))) * (np.e**(v) + (2*v*np.e**(-u))) )

#gradiente em relação a variavel v da nossa função de erro
def gradiente_v(u,v):
    #return ( 2*np.e**(-2*u) * (u*np.e**(u+v) - 2) * (u*np.e**(u+v) - 2*v) )
    return ( 2 * ((u*np.e**(v)) - (2*np.e**(-u))) * (u*np.e**(v) - (2*v*np.e**(-u))) )

#calcula os gradientes direcionais no ponto atual, e só atualiza o ponto depois das duas operações serem feitas
def walk_through_gradient(point, learning_rate):
    direction_u=gradiente_u(point[0], point[1])*learning_rate
    direction_v=gradiente_v(point[0], point[1])*learning_rate
    point[0] -= direction_u
    point[1] -= direction_v

#calcula o gradiente descendente de um função dado um ponto de partida
def descendent_gradient(inicial_point, precision, interations, learning_rate):
    #laço só para depois que é atingido o erro minimo estipulado pelo enunciado
    while (error_function(inicial_point[0], inicial_point[1]) > precision):
        #faz a atualização dos pesos
        walk_through_gradient(inicial_point, learning_rate)
        interations+=1
        #print(error_function(inicial_point[0], inicial_point[1]))
    return interations

#calcula os gradientes direcionais no ponto atual, e atualiza o ponto depois de fazer o gradiente para depois calcular o gradiente na nova direção
def walk_through_coordenate_gradient(point, learning_rate):
    point[0] -= gradiente_u(point[0], point[1])*learning_rate
    point[1] -= gradiente_v(point[0], point[1])*learning_rate

#calcula a função de coordenada descendente a partir de um ponto de partida
def descendent_coordenate(inicial_point, precision, learning_rate):
    i=0
    #laço de apenas 15 iterações assim como é mandado no enunciado
    while i<15:
        walk_through_coordenate_gradient(inicial_point, learning_rate)
        i+=1
        #print(error_function(inicial_point[0], inicial_point[1]))


if __name__ == '__main__':
    #inicialização do ponto inicial, do learning rate e da precisão que o algoritmo precisa ter para sair do laço while do gradiente
    inicial_point=np.array([np.float64(1.),np.float64(1.)])
    learning_rate=np.float64(0.1)
    precision=np.float64(10**(-14))
    interations=0

    #execução do gradiente descendente
    interations=descendent_gradient(inicial_point, precision, interations, learning_rate)

    print("exercicio 11")
    print ("gradient descendent answers:")
    print("interations: ", interations)
    #resposta ex 11 = D
    print("final point: ", inicial_point)
    #resposta ex 12 = E

    print("")

    #execução da coordenada descendente
    inicial_point=np.array([np.float64(1.),np.float64(1.)])

    descendent_coordenate(inicial_point, precision, learning_rate)

    print("exercicio 12")
    print ("coordenate descendent answers:")
    print("final point: ", inicial_point)
    print("error: ", error_function(inicial_point[0], inicial_point[1]))
    #resposta ex 13 = A
