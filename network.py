import sys as sy
import numpy as np
import random as rd
class Neural(object):

    def __init__(self,layers):
        self.layers=layers
        self.numLayers=len(layers)
        self.bias=[np.random.rand(y,1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def feed(self,a):
        for w, b in zip(self.weights,self.bias):
            a=(sigmoid(np.dot(w,a)+b))
        return a


    def consDeri(self,out,y):
        return out-y
    

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDer(x):
    return sigmoid(x)*(1-sigmoid(x))

print Neural([6,7]).bias