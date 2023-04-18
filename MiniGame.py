import random
import pandas as pd
from RLAgent import Agent, RandomAgent

class Deck:
    def __init__(self, deck_size = 13, nb_colors = 4):
        self.cards = [i for i in range(deck_size) for _ in range(nb_colors)]
        random.shuffle(self.cards)

    def pick_card(self, nb_cards = 1):
        return self.cards.pop(0)
    

class Minigame:
    def __init__(self, agent, number_of_games, record_history = False):
        self.possible_actions = [True, False]
        self.state_size = 1
        self.number_of_games = number_of_games
        self.history = []
        self.record_history = record_history

        if self.record_history:
            self.agent = RandomAgent()
        else:
            self.agent = agent
        self.initialization()

    def initialization(self):
        self.agent.initialize(self.possible_actions, self.state_size)
        for i in range(self.number_of_games):
            self.deck = Deck()
            self.score = 0
            self.state = self.deck.pick_card()
            self.loop()

        if self.record_history:
            df = pd.DataFrame(self.history)
            df.to_csv("generatedMinigameData.csv")
        self.test()
    
    def loop(self):
        counter = 0
        while self.score < 10 and len(self.deck.cards) > 0:
            action = self.agent.act(self.state)
            reward = self.react(action)
            self.agent.feedback(reward)

            self.history += [[counter, self.state, action, reward]] 
            counter += 1

    def react(self, action):
        reward = 0
        picked = self.deck.pick_card()
        
        if (picked > self.state and action) or (picked < self.state and not action):
            reward = 1
        else: 
            reward = 0
        self.state = picked
        self.score += reward
        return reward

    def test(self):
        self.deck = Deck()
        for i in range(len(self.deck.cards)):
            last_state = self.state
            print()
            action = self.agent.act(self.state, act_greedy = True)
            reward = self.react(action)
            print('Test: {}, Action: {}, Card: {}, Picked: {}, Reward: {}'.format(i, action, last_state, self.state, reward))


        
if __name__=='__main__':
    agent = Agent()
    game = Minigame(agent, number_of_games= 10000, record_history=False)
    #print('Final Action: {}'.format(action))
