""" la classe de l'environement"""

"""Cette partie represent la classe de l'environement dans laquelle notre agent va faire une action pris dans ma class Agent.
Il fait son observation, il passe à l'étape suivant et reçois la recompence associé à la décision choisi dans son état.
La fonction objectif est diminuer le cout d'achat d'électricité sur le réseau publique toute en respectant les contraints opérationnelles du système.
Les contraints sont: 1- Eviter l'injecter énergie sur le réseau (Cinject_grid) quand microgrid est connecté au réseau
                     2- Eviter de Vider la batterie (Ccons_unsatisfied) quand le système est isolé
                     3- Eviter de faire le shedding de production solaire (Cbatful) quand le système est isolé 

"""
import numpy as np
from actions import *

class MicrogridSimulator:

    def __init__(self, dP, n_SOC, n_pmax, Cbatful, Cinject_grid, Cbat_empty , Cgrid_use  , Cbat_use ):

        self.Cgrid_use =Cgrid_use 
        self.Cbat_use=Cbat_use
        self.dP = dP                                          # Power increment
        self.Cbatful = Cbatful                                # Penalty coefficient
        self.Cinject_grid = Cinject_grid                      # Penalty coefficient
        self.Cbat_empty = Cbat_empty                          # Penalty coefficient
        self.Pgrid=0
        self.n_SOC = n_SOC                                    # number of battery states
        self.SOC = np.arange(self.n_SOC)*dP
        
        self.n_Pnet = 2*n_pmax+1                              # number of power levels (power ranges between -n_power*dP and +n_power*dP)
        self.Pnet = np.arange(-n_pmax,n_pmax+1)*dP
        
        self.n_state = self.n_SOC*self.n_Pnet
        self.state_SOC = np.zeros(self.n_state)
        self.state_Pnet = np.zeros(self.n_state)
       
        state = 0
        for i in range(self.n_SOC) :
            for j in range(self.n_Pnet) :
                self.state_SOC[state] = self.SOC[i]
                self.state_Pnet[state] = self.Pnet[j]
                state += 1
        print(self.state_SOC)
        print(self.state_Pnet)
                
        i_ini, j_ini = self.n_SOC//2, n_pmax
        self.state_ini = self.n_Pnet*i_ini + j_ini
              
        self.state = self.state_ini  # Point de départ 
        self.i = i_ini
        self.j = j_ini
#        i_ini, j_ini= 0, 0
#        self.state_ini=0
 
##################################################################################################################################       
# Verifier les contraints et recevoir le pénalité et la nouvel situation de l'agent: 

    def take_action(self, action, Pnet1):
       
        SOC = self.state_SOC[self.state]                     # Current SOC
        reward = 100
        Pgrid=0
        Pprod_shed=0
        Pcons_unsatisfied=0
        if action == GRID_ON :                               # Grid connexion ON
            SOC1 = SOC                                       # Battery SOC unchanged
            Pgrid = self.state_Pnet[self.state]
            
            reward = -self.Cgrid_use *Pgrid
            # A prévoir ici : reward fonction de Pgrid (penalité si Pgrid<0)
            if Pgrid<0 : 
                reward = self.Cinject_grid*Pgrid            #(negative value)
                Pprod_shed = Pgrid
                Pgrid=0
            
        elif action == GRID_OFF:                            # Grid connexion OFF
#            Pbatt = Pnet1 
            Pbatt = self.state_Pnet[self.state]             # Charge or discharge battery according to the sign of Pnet
            SOC1 = SOC - Pbatt                              # e niveau de charge de la batterie
            reward = -self.Cbat_use*Pbatt                              
               
            # A prévoir ici : reward fonction for Pprod_shed               
            if  SOC1 > self.SOC[-1] :                       # Battery too full, cannot absorb Pnet
                Pnet_actual = SOC1 - self.SOC[-1]           # Pnet actually stored
                SOC1 = self.SOC[-1]                         # Battery state of charge limitation
                Pprod_shed = - Pnet_actual                  # Loss of production (negative value)
                reward = self.Cbatful*Pprod_shed
                
            # A prévoir ici : reward fonction de Pnet_unsatisfied
            if SOC1 < self.SOC[0] :                        # Not enough energy in the battery to provide Pnet
                Pnet_actual = SOC1 - self.SOC[0]           # Pnet actually need 
                SOC1 = self.SOC[0]                         # Battery state of charge limitation
                Pcons_unsatisfied = Pnet_actual            # Unsatisfied consumption (negative value)
                reward = self.Cbat_empty*Pcons_unsatisfied

        state1 = self.set_state(SOC1,Pnet1)
        return state1, reward, Pgrid, Pprod_shed, Pcons_unsatisfied
#################################################################################################################################
# Set de l'indice de l'état:
    def set_k(self, state) :
        if state < 0 :
            print("Index d'état négatif, impossible")
            return -1
        if state>= self.n_state :
            print("Index d'état trop grand, impossible")
            return -1
        self.state = state
        return 0
#################################################################################################################################
# Set  de l'état:       
    def set_state(self, SOC, Pnet) :
        if (SOC<self.SOC[0] or SOC>self.SOC[-1]) :
            print ("Valeur de SOC hors limites")
            return -1
        if (Pnet<self.Pnet[0] or Pnet>self.Pnet[-1]) :
            print ("Valeur de Pnet hors limites")
            return -1
        i, j = int(SOC)//self.dP, int(Pnet)//self.dP+(self.n_Pnet-1)//2
        self.state = self.n_Pnet*i + j
#        print("i=",i," - j=",j," - state=",self.state)
        return self.state

################################################################################################################################
# Reset de l'agent à l'état initial:
    def reset(self):
        self.state = self.state_ini  # Reset state to zero, the beginning of dungeon
        return self.state
    
################################################################################################################################    
    def __str__(self) :
        message = 'Etat courant = ' + str(self.state) + '\n'
        i = self.state // self.n_Pnet
        message += 'SOC = ' + str(self.SOC[i]) + '\n'
        j = self.state % self.n_Pnet
        message += 'Pnet = ' + str(self.Pnet[j]) + '\n'
        return message