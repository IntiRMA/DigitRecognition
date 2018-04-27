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

    def gadientDecent(self, trainingData, runs, batchSize, rate, testData=None):
        if(testData):
            nTest=len(testData)
        n=len(trainingData)
        for i in xrange(runs):
            rd.shuffle(trainingData)
            mini_batches = [
                trainingData[k:k + batchSize]
                for k in xrange(0, n, batchSize)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, rate)
            if testData:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(testData), nTest)
            else:
                print "Epoch {0} complete".format(i)


    def backProp(self,x,y):
        gradientW=[np.zeros(w.shapes) for w in self.weights]
        gradientB=[np.zeros(b.shapes) for b in self.bias]
        activation = x
        activations = [x]
        zs=[]
        for b,w in zip(self.bias,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        delta = self.consDeri(activations[-1], y) * sigmoidDer(zs[-1])
        gradientB[-1]=delta
        gradientW[-1]=np.dot(delta,activations[-2].transpose())
        for l in range(2,self.numLayers):
            z=zs[-l]
            sp=sigmoidDer(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradientB[-l]=delta
            gradientW[-l]=np.dot(delta, activations[-l-1].transpose())
        return (gradientB,gradientW)


    def feed(self,a):
        for w, b in zip(self.weights,self.bias):
            a=(sigmoid(np.dot(w,a)+b))
        return a


    def consDeri(self,out,y):
        return out-y


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidDer(x):
    return sigmoid(x)*(1.0-sigmoid(x))

print Neural([6,7]).bias