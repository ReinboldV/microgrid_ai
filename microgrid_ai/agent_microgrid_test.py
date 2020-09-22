# -*- coding: utf-8 -*-
"""
Dans cette parti on présent notre agent et on construit la Q-Table :
    Au debut on initialise le Q-table à zero et on commance par exploration dans l'espace l'état-action en utilisant la (random_action) 
    au fur et à mesure on pass par l'exploitation des résultats obtenu dans notre Q-Table en utilisant la(greedy_action).
 
"""

import random

GRID_OFF = 0
GRID_ON = 1

#%%
class Agent:
    def __init__(self, q_table):

        self.q_table = q_table
#        print(self.q_table)
###################################################################################################################
#%% Choisir le mode esploration ou exploitation:
    def get_next_action(self, state):
        return self.greedy_action(state)
###################################################################################################################  
#%% Le mode exploitation:
    def greedy_action(self, state):
        # Is GRID_ON  reward is bigger?
        if self.q_table.loc[state , 'GRID_ON'] > self.q_table.loc[state , 'GRID_OFF']:
            return GRID_ON 
        # Is GRID_OFF reward is bigger?
        elif self.q_table.loc[state , 'GRID_OFF'] > self.q_table.loc[state , 'GRID_ON' ]:
            return GRID_OFF
        # Rewards are equal, take random action
        return self.random_action()
#################################################################################################################
#%% Le mode d'exploration:
    def random_action(self):
        if random.random() < 0.5:
            return GRID_ON
        else:  
            return GRID_OFF
        
        
        