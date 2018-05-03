import sys as sy
import numpy as np
import random as rd
class Neural(object):

    def __init__(self,layers):
        #layers of the net
        self.layers=layers
        #numbers of layers in the net
        self.numLayers=len(layers)
        #creates the biases and weights randomly in a gaussian distribution
        self.bias=[np.random.rand(y,1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    """decreases the gradient using a minibatch approche """
    def gadientDecent(self, trainingData, runs, batchSize, rate, testData=None):
        #if there is testData record its length
        if(testData):
            nTest=len(testData)
        #lenght of training
        n=len(trainingData)
        #for each run
        for i in range(runs):
            #shuffle the data and create mini batches
            rd.shuffle(trainingData)
            mini_batches = [
                trainingData[k:k + batchSize]
                for k in range(0, n, batchSize)]
            #update the weightsh based on the minibatch
            for mini_batch in mini_batches:
                self.update(mini_batch, rate)
            #if there is testdata print test out else training
            if testData:
                print ("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(testData), nTest))
            else:
                print ("Epoch {0} complete".format(i))

    """Back propagation algorithm"""
    def backProp(self,x,y):
        #Gradients are started as 0
        gradientW=[np.zeros(w.shapes) for w in self.weights]
        gradientB=[np.zeros(b.shapes) for b in self.bias]
        #list of activations
        activation = x
        activations = [x]
        #stores the weighted inputs
        zs=[]
        #for each weight and bias
        for b,w in zip(self.bias,self.weights):
            #stores the weighted input from all activations from the previouse layer
            #plus the bias of the current perceptron
            z=np.dot(w,activation)+b
            zs.append(z)
            #scales the weighted activation and attaches it to the activations
            activation=sigmoid(z)
            activations.append(activation)
        #this calculates error
        delta = self.constDeri(activations[-1], y) * sigmoidDer(zs[-1])
        #sets the last bias to the error
        gradientB[-1]=delta
        #sets the weight error
        gradientW[-1]=np.dot(delta,activations[-2].transpose())
        #for each of the layers
        for l in range(2,self.numLayers):
            #calculate the error for each layer
            z=zs[-l]
            sp=sigmoidDer(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradientB[-l]=delta
            gradientW[-l]=np.dot(delta, activations[-l-1].transpose())
        return (gradientB,gradientW)

    #Updates the weights using the mini batch approche
    def update(self,miniBatch,rate):
        #starts gradients at zero
        gradientW=[np.zeros(w.shapes) for w in self.weights]
        gradientB=[np.zeros(b.shapes) for b in self.bias]
        #for each minibatch
        for (x,y) in miniBatch:
            #calculate the error
            dGradientB,dGradientW=self.backProp(x,y)
            gradientB=[b+nb for (nb,b) in zip(dGradientB,gradientB) ]
            gradientW=[w+nw for (nw,w) in zip(dGradientW,gradientW)]
            #update weights and biases by that error
            self.weights = [w- (rate/len(miniBatch))*nw for w,nw in zip (self.weights,gradientW)]
            self.bias=[b- (rate/len(miniBatch))*nb for b,nb in zip(self.bias,gradientB)]

    def evaluate(self,tests):
        #evaluates all test inputs
        results=[np.argmax(self.feed(x),y) for x,y in tests]
        return sum(int(x==y) for (x,y) in results)

    def feed(self,a):
        #feeds forward
        for w, b in zip(self.weights,self.bias):
            a=(sigmoid(np.dot(w,a)+b))
        return a


    def constDeri(self, out, y):
        return out-y


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidDer(x):
    return sigmoid(x)*(1.0-sigmoid(x))