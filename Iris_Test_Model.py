from DLfromScratch.Layers import Dense
from DLfromScratch.Activations import LeakyRelu, Sigmoid, Softmax
from DLfromScratch.Model import Model, Cost

from Iris_Data_Preparation import Iris_data


def create_model(input_dim, nb_examples, output_dim):
    ## X.shape must be (nb_features, nb_examples)
    ## Y.shape must be (classes, nb_examples)
    
    model = Model(input_dim, nb_examples)
    model.add(Dense(64, LeakyRelu))
    model.add(Dense(16, Sigmoid))
    #model.add(Dense(Y.shape[0], LeakyRelu))
    model.add(Dense(output_dim, Softmax))
    model.cost_func = Cost.crossentropy
    return model


if __name__=='__main__':
    iris = Iris_data()
    X = iris.X
    Y = iris.Y

    model = create_model(X.shape[0], X.shape[1], Y.shape[0])
    final_cost = model.train(X, Y, 2000)