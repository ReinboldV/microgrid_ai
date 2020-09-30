""" la classe de l'environement"""

"""Cette partie represent la classe de l'environement dans laquelle notre agent va faire une action pris dans ma class Agent.
Il fait son observation, il passe à l'étape suivant et reçois la recompence associé à la décision choisi dans son état.
La fonction objectif est diminuer le cout d'achat d'électricité sur le réseau publique toute en respectant les contraints opérationnelles du système.
Les contraints sont: 1- Eviter l'injecter énergie sur le réseau (Cinject_grid) quand microgrid est connecté au réseau
                     2- Eviter de Vider la batterie (Ccons_unsatisfied) quand le système est isolé
                     3- Eviter de faire le shedding de production solaire (Cbatful) quand le système est isolé 

"""
import numpy as np
import pandas as pd
GRID_OFF = 0
GRID_ON = 1

#%% 
class MicrogridSimulator:

    def __init__(self,n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC,n_Pnet,n_SoC, dp, SoC_min, Pnet_min, dt):
        

        self.n_Pnet = n_Pnet
        self.n_SoC = n_SoC
        self.dp = dp
        self.dt = dt
        self.Pnet_min = Pnet_min
        self.SoC_min = SoC_min        
        self.SoC = SoC
        self.Pnet = Pnet  
        self.SOC_ini = SOC_ini
        self.Pnet_ini = Pnet_ini
        self.Temp_ini = Temp_ini
        self.actions = {0: 'GRID_OFF', 1: 'GRID_ON'}
        self.n_Time = n_Time
#        self.Time = self.dt * np.arange(self.n_Time)
        self.Time = np.arange(self.n_Time)
        SoC2 = np.repeat(self.SoC, len(self.Pnet))
        self.state_SOC = pd.Series(SoC2)


        i_time = int(round(self.Temp_ini / self.dt))
        i_soc  = int(round((self.SOC_ini - self.SoC_min) / self.dp))
        i_Pnet = int(round((self.Pnet_ini - self.Pnet_min) / self.dp))
        
        self.index_ini = self.n_Pnet * (i_time * self.n_SoC + i_soc) + i_Pnet
        
        self.index_state = self.index_ini 
##################################################################################################################################       
#%% Verifier les contraints et recevoir le pénalité et la nouvel situation de l'agent: 

    def take_action(self, action, Pnet1):

        i_Pnet2 = self.index_state  % self.n_Pnet
        residu = self.index_state // self.n_Pnet
        i_soc2 = residu % self.n_SoC
        i_time2 = residu // self.n_SoC
        T = self.Time[i_time2]
        SOC = self.SoC[i_soc2]    # Return the Current SOC
        
        Pgrid = 0
        Pprod_shed = 0
        Pcons_unsatisfied = 0
        
        if action == GRID_ON :                               # Grid connexion ON
            SOC1 = SOC                                       # Battery SOC unchanged
            Pgrid = self.Pnet[i_Pnet2]
            # A prévoir ici : reward fonction de Pgrid (penalité si Pgrid<0)
            if Pgrid<0 : 
                Pprod_shed = Pgrid
                Pgrid = 0
            
        elif action == GRID_OFF:                            # Grid connexion OFF
            Pbatt = self.Pnet[i_Pnet2]                                      # Charge or discharge battery according to the sign of Pnet
            SOC1 = SOC - Pbatt                              # Le niveau de charge de la batterie                             
               
            # A prévoir ici : reward fonction for Pprod_shed               
            if  SOC1 > self.SoC[-1] :                       # Battery too full, cannot absorb Pnet
                Pnet_actual = SOC1 - self.SoC[-1]           # Pnet actually stored
                SOC1 = self.SoC[-1]                         # Battery state of charge limitation
                Pprod_shed = - Pnet_actual                  # Loss of production (negative value)
                
            # A prévoir ici : reward fonction de Pnet_unsatisfied
            if SOC1 < self.SoC[0] :                        # Not enough energy in the battery to provide Pnet
                Pnet_actual = SOC1 - self.SoC[0]           # Pnet actually need 
                SOC1 = self.SoC[0]                         # Battery state of charge limitation
                Pcons_unsatisfied = Pnet_actual            # Unsatisfied consumption (negative value)
            elif action == 'fiftyfifty':
                pass
                
#        T = (T+1)//self.n_Time
#        
#        index_state_1 = self.set_state ( SOC1 , Pnet1, T )
#        
#        return index_state_1, Pgrid, Pprod_shed, Pcons_unsatisfied
                
        T = (T+1) % self.n_Time
        
        index_state = self.set_state ( SOC1 , Pnet1, T )
        
        self.index_state = index_state
        
        return index_state, Pgrid, Pprod_shed, Pcons_unsatisfied
#################################################################################################################################
#%% Set  de l'état:       
    def set_state(self, SOC, Pnet, T) :
        if (SOC<self.SoC[0] or SOC>self.SoC[-1]) :
            print ("Valeur de SOC hors limites")
            return -1
        if (Pnet<self.Pnet[0] or Pnet>self.Pnet[-1]) :
            print ("Valeur de Pnet hors limites")
            return -1

#        i_time = int(round(T/self.dt))
        i_time = int(T)
        i_soc  = int(round((SOC-self.SoC_min)/self.dp))
        i_Pnet = int(round((Pnet-self.Pnet_min)/self.dp))
        self.index_state = self.n_Pnet * (i_time*self.n_SoC + i_soc) + i_Pnet
        return self.index_state         
    
    