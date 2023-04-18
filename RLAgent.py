import random
from copy import deepcopy
import numpy as np

from DLfromScratch.Layers import Dense
from DLfromScratch.Activations import LeakyRelu, Sigmoid, Softmax
from DLfromScratch.Model import Model, Cost


class RandomAgent:
    def __init__(self):
        None

    def initialize(self, possible_actions, *args, **kwargs):
        self.possible_actions = possible_actions

    def act(self, *args, **kwargs):
        return random.choice(self.possible_actions)
    
    def feedback(self, *args, **kwargs):
        None



class Agent:
    def __init__(self):
        self.training_frequency = 16
        self.iterations = 1
        self.batch_size = 16
        self.history = []
        self.history_max_size = 1000

        self.e_greedy_min = 0.1
        self.e_greedy_max = 0.9

        self.e_greedy = self.e_greedy_min
        self.training_counter = 0

    def initialize(self, possible_actions, state_size):
        self.input_dim = state_size + 1
        self.possible_actions = possible_actions
        self.output_dim = len(possible_actions) - 1
        self.model = self.create_model(self.input_dim, self.output_dim, self.batch_size)

    def create_model(self, input_dim, output_dim, batch_size):
        learning_rate = 0.01
        cost_func = Cost.mean_squared_error
        model = Model(input_dim, batch_size, learning_rate, cost_func, verbose = False)
        model.add(Dense(8, LeakyRelu))
        model.add(Dense(output_dim, LeakyRelu))
        return model

    def act(self, state, act_greedy = False):
        self.last_state = state
        if random.random()>self.e_greedy and not act_greedy:
            action = random.randint(0, self.output_dim)
        else:
            action = np.argmax([self.model.predict(np.array([[state]+[action_example]]).T) for action_example in [0, 1]])
        self.last_action = action
        return self.possible_actions[action]    
        
    def feedback(self, reward):
        if len(self.history) > self.history_max_size:
            del self.history[0]   
        if self.training_counter > self.training_frequency:
            self.training_counter = 0
            self.train_model()
        else:
            self.training_counter +=1
        self.history += [[self.last_state, self.last_action, reward]] 

    def train_model(self):
        training_data = random.choices(self.history, k=self.batch_size)
        X = np.array([[elem[0],elem[1]] for elem in training_data]).T
        Y = np.array([elem[2] for elem in training_data]).reshape(len(training_data), 1).T
        
        self.model.train(X, Y, self.iterations)