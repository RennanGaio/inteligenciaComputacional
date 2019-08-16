# -*- coding: utf-8 -*-
"""
Aluno: Rennan de Lucena Gaio
Codigo referente aos 4 exercícios 11 e 12 da lista 1 de IC2
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def error_function(u,v):
    return ( u*np.e**(v) - 2*v*np.e**(-u) )**2

def gradiente_u(u,v):
    return ( 2 * ((u*np.e**(v)) - (2*v*np.e**(-u))) * (np.e**(v) + (2*v*np.e**(-u))) )

def gradiente_v(u,v):
    #return ( 2*np.e**(-2*u) * (u*np.e**(u+v) - 2) * (u*np.e**(u+v) - 2*v) )
    return ( 2 * ((u*np.e**(v)) - (2*np.e**(-u))) * (u*np.e**(v) - (2*v*np.e**(-u))) )

def walk_through_gradient(point, learning_rate):
    direction_u=gradiente_u(point[0], point[1])*learning_rate
    direction_v=gradiente_v(point[0], point[1])*learning_rate
    point[0] -= direction_u
    point[1] -= direction_v

def descendent_gradient(inicial_point, precision, interations, learning_rate):
    while (error_function(inicial_point[0], inicial_point[1]) > precision):
        walk_through_gradient(inicial_point, learning_rate)
        interations+=1
        #print(error_function(inicial_point[0], inicial_point[1]))
    return interations

def walk_through_coordenate_gradient(point, learning_rate):
    point[0] -= gradiente_u(point[0], point[1])*learning_rate
    point[1] -= gradiente_v(point[0], point[1])*learning_rate

def descendent_coordenate(inicial_point, precision, learning_rate):
    i=0
    while i<15:
        walk_through_coordenate_gradient(inicial_point, learning_rate)
        i+=1
        #print(error_function(inicial_point[0], inicial_point[1]))


if __name__ == '__main__':
    inicial_point=np.array([np.float64(1.),np.float64(1.)])
    learning_rate=np.float64(0.1)
    precision=np.float64(10**(-14))
    interations=0

    interations=descendent_gradient(inicial_point, precision, interations, learning_rate)

    print ("gradient descendent answers:")
    print("interations: ", interations)
    #resposta ex 11 = D
    print("final point: ", inicial_point)
    #resposta ex 12 = E

    print("")

    inicial_point=np.array([np.float64(1.),np.float64(1.)])

    descendent_coordenate(inicial_point, precision, learning_rate)

    print ("coordenate descendent answers:")
    print("final point: ", inicial_point)
    print("error: ", error_function(inicial_point[0], inicial_point[1]))
    #resposta ex 13 = A
