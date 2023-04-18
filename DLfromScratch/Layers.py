import numpy as np

class Weight_Init:
    def random_init(layer_size, input_size):
        w = np.random.randn(layer_size, input_size)*0.01
        b = np.zeros(shape=(layer_size, 1))
        return w, b

class Dense:
    def __init__(self, layer_size, activation):
        self.layer_size = layer_size
        self.activation = activation
        
    def activate(self, input_dim, learning_rate):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.W, self.B = Weight_Init.random_init(self.layer_size, input_dim)
        self.adam_initialization()
    
    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, X)+self.B
        self.A = self.activation.classic(self.Z)
        return self.A
    
    def backwards(self, dA_last, m_examples, it):
        if self.activation.__name__ == 'Softmax':
            dZ = dA_last
        else:
            dZ = dA_last * self.activation.derivative(self.Z)
        dW = np.dot(dZ, self.X.T)/m_examples
        dB = np.expand_dims(1/m_examples * np.sum(dZ, axis=1), axis=1)
        dW, dB = self.adam_optimization(dW, dB, it)
        self.update_weights(dW, dB)
        dA_last = np.dot(self.W.T, dZ)
        return dA_last
    
    def update_weights(self, dW, dB):
        self.W -= self.learning_rate * dW
        self.B -= self.learning_rate * dB
        
    def adam_initialization(self):
        self.Beta1 = 0.9
        self.Beta2 = 0.999
        self.epsi = 1e-08
        self.VdW, self.SdW, self.VdB, self.SdB = 0, 0, 0, 0
        
    def adam_optimization(self, dW, dB, it): ##A revérifier
        self.VdW = self.Beta1*self.VdW + (1-self.Beta1)*dW
        self.VdB = self.Beta1*self.VdB + (1-self.Beta1)*dB
        self.SdW = self.Beta2*self.SdW + (1-self.Beta2)*dW*dW
        self.SdB = self.Beta2*self.SdB + (1-self.Beta2)*dB*dB
        VdWcorr = self.VdW/(1-self.Beta1**it)
        VdBcorr = self.VdB/(1-self.Beta1**it)
        SdWcorr = self.SdW/(1-self.Beta2**it)
        SdBcorr = self.SdB/(1-self.Beta2**it)
        dW_to_return = VdWcorr/(np.sqrt(SdWcorr)+self.epsi)
        dB_to_return = VdBcorr/(np.sqrt(SdBcorr)+self.epsi)
        return dW_to_return, dB_to_return