# -*- coding: utf-8 -*-
"""
Dans cette parti on présent notre agent et on construit la Q-Table :
    Au debut on initialise le Q-table à zero et on commence par exploration dans l'espace l'état-action en utilisant la (random_action)
    au fur et à mesure on pass par l'exploitation des résultats obtenu dans notre Q-Table en utilisant la(greedy_action).
 
"""
import random

import numpy as np
import pandas as pd

GRID_OFF = 0
GRID_ON = 1


class Agent:
    def __init__(self, n_state, n_episode, learning_rate, discount,
                 exploration_rate, iterations, actions):

        self.n_episode = n_episode
        self.n_state = n_state
        self.actions = actions
        self.q_table = pd.DataFrame(0, index=np.arange(n_state), columns=self.actions.values())
        self.learning_rate = learning_rate  # How much we appreciate new q-value over current
        self.discount = discount  # How much we appreciate future reward over current
        self.iterations = iterations
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations  # Shift from exploration to exploitation

    def get_next_action(self, state, indicat):
        """
        Choix du mode exploration ou exploitation :
        param state:
        return:
        """
        if (indicat == 'Train'):
            
            if random.random() > self.exploration_rate:  # Explore (gamble) or exploit (greedy)
                return self.greedy_action(state)
            else:
                return self.random_action()
        else:
            return self.greedy_action(state)

    def greedy_action(self, state):
        """
        Mode exploitation:
        param : state
        return:
        """
        # TODO: Faire une recherche de MAX dans la q_table à l'état state.
        #  Pour etre généralisable à n'importe
        #  quel nombre d'actions. Ne pas écrire en dur les actions.
        # Is GRID_ON  reward is bigger?
        if self.q_table.loc[state, 'GRID_ON'] > self.q_table.loc[state, 'GRID_OFF']:
            return GRID_ON
        # Is GRID_OFF reward is bigger?
        elif self.q_table.loc[state, 'GRID_OFF'] > self.q_table.loc[state, 'GRID_ON']:
            return GRID_OFF
        # Rewards are equal, take random action
        return self.random_action()

    @staticmethod
    def random_action():
        """
        mode d'exploration
        """

        # TODO: modifier pour avoir un nombre variable d'action possibles (len(actions))
        if random.random() < 0.5:
            return GRID_ON
        else:
            return GRID_OFF

    def update(self, old_state, new_state, action, reward, indicat):
        """
        Update le Q-Table

        :param old_state:
        :param new_state:
        :param action:
        :param reward:
        :return:
        """
        if (indicat == 'Train'): 
            
            old_value = self.q_table.loc[old_state].values[action]  # Old Q-table value
            #        print('q_table = ',self.q_table)
            future_action = self.greedy_action(new_state)  # What would be our best next action?
    
            future_reward = self.q_table.loc[new_state].values[future_action]  # What is reward for the best next action?
    
            # Main Q-table updating algorithm:
            new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
    
            # TODO : modifier pour la colonne action : ['GRID_OFF', 'GRID_ON'][action] - > actions[action]
            self.q_table.loc[old_state, ['GRID_OFF', 'GRID_ON'][action]] = new_value
    
            # Finally, shift our exploration_rate toward zero (less gambling)
            if self.exploration_rate > 0:
                self.exploration_rate -= self.exploration_delta
        else:
            pass