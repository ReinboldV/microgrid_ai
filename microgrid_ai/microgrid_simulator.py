""" class of environmentla """

"""
This part represents the class of the environment in which our agent will do an action taken in my class Agent.
He makes his observation, he goes to the next step and receives the reward associated with the decision chosen in his state.
The objective function is to reduce the cost of purchasing electricity on the public network while respecting the operational constraints of the system.
The constraints are:  1- Avoid injecting energy into the network (Cinject_grid) when microgrid is connected to the network
                      2- Avoid draining the battery (Ccons_unsatisfied) when the system is isolated
                      3- Avoid shedding solar production (Cbatful) when the system is isolated

"""



import numpy as np
import pandas as pd

GRID_OFF = 0
GRID_ON = 1


class MicrogridSimulator:

    def __init__(self, n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC, n_Pnet, n_SoC, dp, SoC_min, Pnet_min,
                 Cbatful, C_PV_shed, Cbat_empty, Cgrid_use_Creuse, Cgrid_use_plaine, Cbat_use, dt):
        """
        Simulation of the microgrids... 

        :param n_Time: Size of the episode
        :param Pnet_ini: Net Power of the first episode
        :param SOC_ini: Initial state of charge for the first episode
        :param Temp_ini: Initial time stamp of the episode
        :param Pnet: DataFrame of the Net Power available
        :param SoC: DataFrame of the SOC available
        :param n_Pnet: Discretisation Number of Not Power states (x)
        :param n_SoC: Discretisation Number of SOC states (x)
        :param dp: Discretisation of Not Power states
        :param SoC_min: Minimal state of the SOC (x)
        :param Pnet_min: Minimal state of the Net Power in the database (x)
        :param dt: Time state (default = 0.5 h)
        :param Cbatful: Penalty for SOC = SOC_max
        :param C_PV_shed: Penalty for PV shading
        :param Cbat_empty: Penalty for SOC = SOC_min
        :param Cgrid_use_Creuse: Low hour cost cost
        :param Cgrid_use_plaine: Peak hour cost
        :param Cbat_use: Penalty for using the battery
        """
        self.Cgrid_use_Creuse = Cgrid_use_Creuse
        self.Cgrid_use_plaine = Cgrid_use_plaine
        self.Cbat_use = Cbat_use  # Penalty coefficient
        self.Cbatful = Cbatful  # Penalty coefficient
        self.C_PV_shed = C_PV_shed  # Penalty coefficient
        self.Cbat_empty = Cbat_empty  # Penalty coefficient

        self.actions = {0: 'GRID_OFF', 1: 'GRID_ON'}

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

        self.n_Time = n_Time
        self.Time = np.arange(self.n_Time)
        SoC2 = np.repeat(self.SoC, len(self.Pnet))
        self.state_SOC = pd.Series(SoC2)

        self.state_Pnet = pd.Series(self.Pnet)
        self.state_Pnet = [self.state_Pnet] * (len(self.SoC))
        self.state_Pnet = pd.concat(self.state_Pnet, ignore_index=True)

        self.env1 = pd.DataFrame({'state_SOC': self.state_SOC, 'state_Pnet': self.state_Pnet})
        self.env1 = [self.env1] * (self.n_Time)
        self.env1 = pd.concat(self.env1, ignore_index=True)

        self.temp1 = np.arange(self.n_Time)
        self.temp1 = np.repeat(self.temp1, (len(self.SoC) * len(self.Pnet)))
        self.temp1 = pd.Series(self.temp1)
        self.temp1 = pd.DataFrame({'temps': self.temp1})

        self.tarif = pd.DataFrame({'Tarif': ['HC', 'HC', 'HC', 'HC', 'HC', 'HC', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP',
                                             'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HC', 'HC']})
        self.tarif = pd.DataFrame(np.repeat(self.tarif.values, int(round(self.n_Time / 24)), axis=0))
        self.tarif = pd.DataFrame(np.repeat(self.tarif.values, (len(self.SoC) * len(self.Pnet)), axis=0))
        self.tarif.columns = ['Tarif']

        self.env = pd.concat([self.temp1, self.env1], axis=1)
        self.env = pd.concat([self.env, self.tarif], axis=1)
        i_time = int(round(self.Temp_ini / self.dt))
        i_soc = int(round((self.SOC_ini - self.SoC_min) / self.dp))
        i_Pnet = int(round((self.Pnet_ini - self.Pnet_min) / self.dp))

        self.index_ini = self.n_Pnet * (i_time * self.n_SoC + i_soc) + i_Pnet

        self.index_state = self.index_ini
        
        
        
        
        
    def calcule_reel_p(self, teau_incertitude, i_Pnet2, Pnet):
        """
         This Function considers the uncertainty of the prediction into count
         and calculates the reel Pnet compared to the predicted Pnet:
         It receives the index from Pnet at each moment and it considers
             a probability of (1 - teau_incertitude)% for the case (prediction is realized in reality)
             a probability of (teau_incertitude / 2)% for the case of under-production
             a probability of (teau_incertitude / 2)% for the case of over-production 
        """
        i_Preel = 0
        x = np.random.random()
        
        if (x > (teau_incertitude/100)):
            i_Preel = i_Pnet2
                    
        elif(((teau_incertitude/100)/2) < x <= (teau_incertitude/100)):
            i_Preel = i_Pnet2 - 1
            if (i_Preel < 0):
                i_Preel = 0 
                
        elif(x <= ((teau_incertitude/100)/2)):
            i_Preel = i_Pnet2 + 1
            if (i_Preel > len(Pnet)-1):
                i_Preel = len(Pnet)-1
                
        P_reel = Pnet [i_Preel]

        return P_reel

    # Check the constraints and receive the penalty and the new state :
    def take_action(self,teau_incertitude, action, pnet):
        """
        :param action: action of the agent at time t
        :param pnet: net power demand at time t
        :return: 
        """
        i_Pnet2 = self.index_state % self.n_Pnet
        residu = self.index_state // self.n_Pnet
        i_soc2 = residu % self.n_SoC
        i_time2 = residu // self.n_SoC
        T = self.Time[i_time2]
        SOC = self.SoC[i_soc2]  # Return the Current SOC

        reward = 100
        Pgrid = 0
        Pprod_shed = 0
        Pcons_unsatisfied = 0
        
        P_reel = self.calcule_reel_p (teau_incertitude, i_Pnet2, self.Pnet)

        if action == GRID_ON:  # Grid connexion ON
            SOC1 = SOC  # Battery SOC unchanged

#            Pgrid = self.Pnet[i_Pnet2]  
            Pgrid = P_reel

            if self.env.at[self.index_state, 'Tarif'] == 'HP':
                reward = -self.Cgrid_use_plaine * Pgrid
            else:
                reward = -self.Cgrid_use_Creuse * Pgrid

            # Reward fonction de Pgrid <0
            if Pgrid < 0:
                reward = self.C_PV_shed * Pgrid  # (negative value)
                Pprod_shed = Pgrid
                Pgrid = 0

        elif action == GRID_OFF:  # Grid connexion OFF

#            Pbatt = self.Pnet[i_Pnet2]  # Charge or discharge battery according to the sign of Pnet
            Pbatt = P_reel
            SOC1 = SOC - Pbatt  # The battery charge level
            reward = -self.Cbat_use * Pbatt

            # A prévoir ici : reward fonction for Pprod_shed               
            if SOC1 > self.SoC[-1]:  # Battery too full, cannot absorb Pnet
                Pnet_actual = SOC1 - self.SoC[-1]  # Pnet actually stored
                SOC1 = self.SoC[-1]  # Battery state of charge limitation
                Pprod_shed = - Pnet_actual  # Loss of production (negative value)
                reward = self.Cbatful * Pprod_shed

            # A prévoir ici : reward fonction de Pnet_unsatisfied
            if SOC1 < self.SoC[0]:  # Not enough energy in the battery to provide Pnet
                Pnet_actual = SOC1 - self.SoC[0]  # Pnet actually need
                SOC1 = self.SoC[0]  # Battery state of charge limitation
                Pcons_unsatisfied = Pnet_actual  # Unsatisfied consumption (negative value)
                reward = self.Cbat_empty * Pcons_unsatisfied

        elif action == 'fiftyfifty':
            pass
        
        T = (T+1) % self.n_Time
        
        index_state = self.set_state ( SOC1 , pnet, T )
        
        self.index_state = index_state
        
        return index_state, reward, P_reel, Pgrid, Pprod_shed, Pcons_unsatisfied

    def set_state(self, SOC, Pnet, T):
        """

        :param SOC:
        :param Pnet:
        :param T:
        :return:
        """
        if SOC < self.SoC[0] or SOC > self.SoC[-1]:
            print("Valeur de SOC hors limites")
            return -1
        if Pnet < self.Pnet[0] or Pnet > self.Pnet[-1]:
            print("Valeur de Pnet hors limites")
            return -1

        i_time = int(T)
        i_soc = int(round((SOC - self.SoC_min) / self.dp))
        i_Pnet = int(round((Pnet - self.Pnet_min) / self.dp))
        self.index_state = self.n_Pnet * (i_time * self.n_SoC + i_soc) + i_Pnet
        return self.index_state
