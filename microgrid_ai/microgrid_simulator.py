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
        self.Time = self.dt * np.arange(self.n_Time)

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

    # Verifier les contraints et recevoir la pénalité et la nouvelle situation de l'agent :
    def take_action(self, action, pnet):
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

        if action == GRID_ON:  # Grid connexion ON
            SOC1 = SOC  # Battery SOC unchanged

            Pgrid = self.Pnet[i_Pnet2]  # = pnet(t-1)

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

            Pbatt = self.Pnet[i_Pnet2]  # Charge or discharge battery according to the sign of Pnet
            SOC1 = SOC - Pbatt  # Le niveau de charge de la batterie
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

        T = (T + 1) // self.n_Time

        index_state_1 = self.set_state(SOC1, pnet, T)

        return index_state_1, reward, Pgrid, Pprod_shed, Pcons_unsatisfied

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

        i_time = int(round(T / self.dt))
        i_soc = int(round((SOC - self.SoC_min) / self.dp))
        i_Pnet = int(round((Pnet - self.Pnet_min) / self.dp))
        self.index_state = self.n_Pnet * (i_time * self.n_SoC + i_soc) + i_Pnet
        return self.index_state

    def get_soc_time_pnet(self):
        i_Pnet2 = self.index_state % self.n_Pnet
        residu = self.index_state // self.n_Pnet
        i_soc2 = residu % self.n_SoC
        i_time2 = residu // self.n_SoC

        time = self.Time[i_time2]
        soc = self.SoC[i_soc2]
        pnet = self.Pnet[i_Pnet2]

        return soc, time, pnet



