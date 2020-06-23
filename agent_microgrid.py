# -*- coding: utf-8 -*-
"""
Dans cette parti on présent notre agent et on construit la Q-Table :
    Au debut on initialise le Q-table à zero et on commance par exploration dans l'espace l'état-action en utilisant la (random_action) 
    au fur et à mesure on pass par l'exploitation des résultats obtenu dans notre Q-Table en utilisant la(greedy_action).
 
"""
from actions import *
import numpy as np
import random


#GRID_OFF = 0
#GRID_ON = 1


class Agent:
    def __init__(self,n_state, n_actions, learning_rate, discount, exploration_rate , iterations):
        self.q_table = np.zeros((n_state, n_actions))                         # Spreadsheet (Q-table) for rewards accounting
        self.learning_rate = learning_rate                      # How much we appreciate new q-value over current
        self.discount = discount                                # How much we appreciate future reward over current
        self.iterations=iterations
        self.exploration_rate = exploration_rate                            # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations               # Shift from exploration to explotation

###################################################################################################################
# Choisir le mode esploration ou exploitation:
    def get_next_action(self, state):
        if random.random() > self.exploration_rate:            # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

###################################################################################################################
# Le mode exploitation:
    def greedy_action(self, state):
        # Is GRID_ON  reward is bigger?
        if self.q_table[state][GRID_ON ] > self.q_table[state][GRID_OFF]:
            return GRID_ON 
        # Is GRID_OFF reward is bigger?
        elif self.q_table[state][GRID_OFF] > self.q_table[state][GRID_ON ]:
            return GRID_OFF
        # Rewards are equal, take random action
        if random.random() < 0.5:
            return GRID_ON 
        else :
           return GRID_OFF
       
#################################################################################################################
# Le mode d'exploration:
    def random_action(self):
        if random.random() < 0.5:
            return GRID_ON
        else:  
            return GRID_OFF

#################################################################################################################
# Update le Q-Table:
    def update(self, old_state, new_state, action, reward):
        
        old_value = self.q_table[old_state][action]                        # Old Q-table value
        
        future_action = self.greedy_action(new_state)                      # What would be our best next action?
 
        future_reward = self.q_table[new_state][future_action]             # What is reward for the best next action?

        # Main Q-table updating algorithm:
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[old_state][action] = new_value
        
        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta