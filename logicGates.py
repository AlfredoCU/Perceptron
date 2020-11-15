#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:28:50 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt

# Neurona artificial.
class Perceptron:
    # Inicializamos nuestros valores random.
    # nInput es la cantidad de patrones.
    # learningRate es nuestro factor de aprendizaje.
    def __init__(self, nInput, learningRate):
        # Valores random para a segurar que no hay un conocimiento previo. 
        self.w = -1 + 2 * np.random.rand(nInput)
        self.b = -1 + 2 * np.random.rand()
        # Se almacena en eta para usar el valor del factor dentro de la clase.
        self.eta = learningRate
    
    
    # Predicciones.
    def predict(self, X):
        # Nos da las dimenciones de la matrix tomamos [1] ya que es la cantidad de patrones.
        p = X.shape[1]
        # Matriz de 0, con el mismo tamaño de w.
        yEst = np.zeros(p)
        for i in range(p):
            # Y estimada es la multimplicacion de w por X (datos) + b
            yEst[i] = np.dot(self.w, X[:,i]) + self.b
            # Resultado de la predicción.
            if yEst[i] >= 0:
                yEst[i] = 1
            else:
                yEst[i] = 0
        return yEst
    
    
    # Entrenamiento Perceptron.
    # X -> DATOS.
    # Y -> APROXIMACION.
    # epoch -> cantidad de iteraciones.
    def fit(self, X, Y, epoch = 50):
        p = X.shape[1]
        # _ ignoramos no necesitamos esa variable.
        for _ in range(epoch):
            # iteramos por patrones.
            for i in range(p):
                # ? X[:, i].reshape(-1, 1)
                yEst = self.predict(X[:, i].reshape(-1, 1))
                # ? X[:, i]
                self.w += self.eta * (Y[i] - yEst) * X[:, i]
                self.b += self.eta * (Y[i] - yEst)
                

# Dibujar la grafica.
def draw2d(model):
    # Valores w, b de la neurona.
    w1, w2, b = model.w[0], model.w[1], model.b
    # Dimenciones.
    li, ls = -2, 2
    # Para hacer la recta.
    plt.plot([li, ls], [(1/w2) * (-w1 * (li) -b), (1/w2) * (-w1 * (ls) -b)], "--k")
    
    
# Instanciamos Perceptron.
neuron = Perceptron(2, 0.1)

# X datos.  
X = np.array([[0.5], [0, 1, 0, 1]])

# Y Valores deseados.
Y = np.array([0, 0, 0, 1]) # COMPUERTA AND.
# Y = np.array([0, 1, 1, 1]) # COMPUERTA OR.
# Y = np.array([0, 1, 1, 0]) # COMPUERTA XOR.

# Entrenamos x sobre y.
neuron.fit(X, Y)

# Predicion.
print(neuron.predict(X))


#Dibujar las compuertas.
_, p = X.shape
for i in range(p):
    if(Y[i] == 0):
        plt.plot(X[0, i], X[1, i], "xr")
    else:
        plt.plot(X[0, i], X[1, i], "ob")
        

plt.title("Perceptrón - Compuerta AND")
plt.grid("on")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

draw2d(neuron)

# Es para guardar una imagen en un formato eps, para mejor calidad de imagen.
plt.savefig('and.eps', format="eps")
