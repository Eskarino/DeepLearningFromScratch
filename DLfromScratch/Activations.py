from scipy.special import expit, softmax
import numpy as np

class Sigmoid:
    def classic(x):
        return expit(x)
    
    def derivative(x):
        return Sigmoid.classic(x)*(1-Sigmoid.classic(x))

class Relu:
    def classic(x):
        return x * (x > 0)

    def derivative(x):
        return 1 * (x > 0)
    
class LeakyRelu:
    def classic(x):
        return x * (x > 0) + x*0.01 * (x < 0)

    def derivative(x):
        return 1 * (x > 0) + 0.01 * (x < 0)
     
class Linear:
    def classic(x):
        return x
    
    def derivative(x):
        return 1 * (x==x)
    
class Softmax:
    def classic(x):
       rez = np.array([softmax(xi) for xi in x.T]).T
       return rez
    
    def derivative(x):
        None ## A definir