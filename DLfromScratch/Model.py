import sys
import numpy as np
from copy import deepcopy

class Cost:
    def mean_absolute_error(y, y_hat):
        return np.abs(y-y_hat)/len(y)
    
    def mean_squared_error(y, y_hat):
        return (y-y_hat)**2/len(y)
    
    def binary_loss(y, y_hat):
        return -(1-y)/(1-y_hat)+(y/y_hat)
    
    def binary_crossentropy(y, y_hat):
        return y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    
    def crossentropy(y, y_hat):
        return (y*np.log(y_hat)).sum()
    
    def cost(func, y, y_hat):
        return -func(y, y_hat).sum()


class Model:
    def __init__(self, input_size, batch_size, learning_rate = 0.01, cost_func = Cost.crossentropy, verbose = True):
        self.layers = []
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_func = cost_func
        self.verbose = verbose
        
    def lr_decay(self, it, lr_decay_rate = 0.0001):
        self.learning_rate /= 1+lr_decay_rate*it
        
    def add(self, layer):
        if self.layers == []:
            input_dim = self.input_size
        else:
            input_dim = self.layers[-1].layer_size
        layer.activate(input_dim, self.learning_rate)
        self.layers += [layer]
        
    def predict(self, X):
        last_activation = X
        for layer in self.layers:
            last_activation = layer.forward(last_activation)
        return last_activation
    
    def backprop(self, last_activation, Y, it):
        dA = last_activation - Y
        for layer in self.layers[::-1]:
            dA = layer.backwards(dA, self.batch_size, it)
            
    def calculate_cost(self, y, y_hat):
        self.cost = Cost.cost(self.cost_func, y, y_hat)
        
    def print_log(self, iteration):
        sys.stdout.flush()
        sys.stdout.write('Iteration: ' + str(iteration+1) + ' - Cost: ' + str(self.cost)+ '\r')
        #print('Iteration: ' + str(iteration+1) + ' - Cost: ' + str(self.cost))
            
    def train(self, X, Y, num_iterations):
        X = deepcopy(X)
        Y = deepcopy(Y)
        for i in range(num_iterations):
            last_activation = self.predict(X)
            self.calculate_cost(Y, last_activation)
            self.backprop(last_activation, Y, i+1)
            if self.verbose:
                self.print_log(i)
        return(self.cost)
            #self.lr_decay(i)

    def view_layer_dims(self):
        for l, layer in enumerate(self.layers):
            print('Layer: {} - Shape: {}'.format(l, layer.W.shape))