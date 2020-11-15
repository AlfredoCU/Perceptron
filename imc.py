#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:48:27 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt

# Neurona artificial.
class Perceptron:
    # Inicializamos nuestros valores random.
    def __init__(self, nInput, learningRate):
        self.w = -1 + 2 * np.random.rand(nInput)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learningRate
    
    
    # Predicciones.
    def predict(self, X):
        p = X.shape[1]
        yEst = np.zeros(p)
        for i in range(p):
            yEst[i] = np.dot(self.w, X[:,i]) + self.b
            if yEst[i] >= 0:
                yEst[i] = 1
            else:
                yEst[i] = 0
        return yEst
    
    
    # Entrenamiento.
    def fit(self, X, Y, epoch = 50):
        p = X.shape[1]
        for _ in range(epoch):
            for i in range(p):
                yEst = self.predict(X[:, i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - yEst) * X[:, i]
                self.b += self.eta * (Y[i] - yEst)
                

# Dibujar la grafica.
def draw2d(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    li, ls = -2, 2
    plt.plot([li, ls], [(1/w2) * (-w1 * (li) -b), (1/w2) * (-w1 * (ls) -b)], "--k")
    
    
# Instanciamos Perceptron.
model = Perceptron(2, 0.1)

# Poblacion.
p = 30

# Crear datos.
X = np.zeros((2, p))
Y = np.zeros(p)

for i in range(p):
    # Masa aleatoria.
    X[0, i] = -40 + (120 + 40) * np.random.rand()
    # Estatura aleatoria.
    X[1 ,i] = -1 + (2.2 + 1) * np.random.rand()
    imc = X[0, i] / X[1, i]**2
    if imc >= 25:
        Y[i] = 1
    else:
        Y[i] = 0
    
    
# Normalizar los datos.
X[0, :] = (X[0, :] - X[0, :].min()) / (X[0, :].max() - X[0, :].min())
X[1, :] = (X[1, :] - X[1, :].min()) / (X[1, :].max() - X[1, :].min())

# Entrenamos x sobre y.
model.fit(X, Y, 100)

# Predicion.
print(model.predict(X))


#Dibujar las compuertas.
for i in range(p):
    if(Y[i] == 0):
        plt.plot(X[0, i], X[1, i], "xr")
    else:
        plt.plot(X[0, i], X[1, i], "ob")
        

plt.title("IMC")
plt.grid("on")
plt.xlim([-0.25, 1.25])
plt.ylim([-0.25, 1.25])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

draw2d(model)

# Es para guardar una imagen en un formato eps, para mejor calidad de imagen.
plt.savefig('imc.eps', format="eps")
