# -*- coding: utf-8 -*-

"""
Created on Fri May 15 10:30:13 2020

@author: mdini
"""
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from microgrid_ai.agent_microgrid_test import Agent
from microgrid_ai.microgrid_simulator_test import MicrogridSimulator
plt.close('all')

# %%
start_day_month_test = 1
n_jour_test = 7

dt = 0.5  # [h] Pas de discrétisation du temps
dp = 1400  # Pas de discrétisation de Puissance
SoC_min, SoC_max = 0, 21000  # [W.h] Capacité min et max de la batterie   

# le tarif bleu du réseau :
C_conso_unsatisfied = 10  # euros par KWh
Cout_grid_plaine = 0.133  # euros par KWh
Cout_grid_Creuse = 0.11  # euros par KWh

# %% Récupérer les données historiques de Pnet = Pload - Ppv  (Toutes les 30 minutes) :
"""
On prend une sample avec (le nombre de jour = n_episode) de notre data_base
Chaque épisode présente un jour choisi par hazard dans notre database

"""
data_path = r"C:\Users\mdini\Documents\GitHub\microgrid_ai\data\drahix"

month = pd.Series(pd.date_range('8/1/2018', freq='D', periods=31))

data_base = pd.DataFrame({'Dates': month})
data_base['Dates'] = data_base['Dates'].astype(str)

#chosen_idx = np.random.choice(len(data_base), replace=True, size = n_jour_test)           # Choose days randomely
chosen_idx = np.arange(start_day_month_test -1 , start_day_month_test + n_jour_test-1)     # Choose days by roders

sample_data_names = data_base.iloc[chosen_idx]

sample_data_names = 'DrahiX_5min_to_30Min@' + sample_data_names + '.txt'
sample_data_names = sample_data_names.set_index((np.arange(n_jour_test)))

Testing_data = pd.read_csv(os.path.join(data_path, str(sample_data_names.Dates[0])), delimiter="\t")
Testing_data = Testing_data.drop(Testing_data.index[len(Testing_data) - 1])

for i in range(1, n_jour_test):
    X = pd.read_csv(os.path.join(data_path, str(sample_data_names.Dates[i])), delimiter="\t")
    X = X.drop(X.index[len(X) - 1])
    Testing_data = pd.concat([Testing_data, X], axis=0, join='inner', ignore_index=True)

Pnet1 = ((Testing_data.Cons - Testing_data.Prod) // dp) * dp

#%% Récuperer la Q_Table obtenue par les données historiques
file = 'Q_table_2mois_centmills'
file_read = file + '.txt'
q_table = pd.read_csv(os.path.join(data_path, file_read), delimiter="\t")
####################################################################################################################  
#%%
t0 = time.time()
# %% Initialisation de l'environement et l'espace d'état:
n_points = len(Pnet1)
Temp_ini = 0
Pnet_ini = Pnet1[0]
Pnet_min, Pnet_max = min(Pnet1),max(Pnet1)  # [W]
n_Time = int(round(24./dt))
n_Pnet = int((Pnet_max - Pnet_min)/dp+1)
n_SOC = int((SoC_max/dp)+1)
n_state = n_SOC*n_Pnet*n_Time                     # Initialisation de la taille de l'espace d'état 

# SOC array
SoC = np.linspace(SoC_min,SoC_max,n_SOC)

SOC_ini = 10500                                  # Choisir l'état initiale manuele
#SOC_ini =int(np.random.choice(SoC))               # Choisir l'état initiale par hazard 

# Pnet array
Pnet = np.linspace(Pnet_min,Pnet_max,n_Pnet)
###############################################################################################################
#%% Construire de l'environement et l'agent:
microgrid = MicrogridSimulator(n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC,n_Pnet,n_SOC, dp, SoC_min, Pnet_min, dt)

agent = Agent(q_table)
################################################################################################################  
#%% main loop:    

### Initialisation des paramètres ###

Pgrid_step = np.zeros(n_points)
Pprod_shed_step=np.zeros(n_points)
Pcons_unsatisfied_step=np.zeros(n_points)
Statofcharge=np.zeros(n_points)

### Coeur de l'algorithme ###

                                 
for step in range(n_points):
    
    old_state = microgrid.index_state

    action = agent.get_next_action(old_state)                     # Query agent for the next action

    new_state, Pgrid, Pprod_shed, Pcons_unsatisfied = microgrid.take_action(action,Pnet1[step]) # Take action, get new state and reward

    Statofcharge[step] = microgrid.state_SOC[new_state] 
#    Statofcharge[step]  = microgrid.env.loc[new_state,'state_SOC']
    Pgrid_step[step] = Pgrid
    
    Pprod_shed_step[step] = Pprod_shed
    
    Pcons_unsatisfied_step[step] = Pcons_unsatisfied
#################################################################################################        
# %% Calcule de Coût :
"""
Calculer le cout en considérant le tarif bleu 
Heures creuses = 22h jusqu'à 6h
Heures plaines = 6h jusqu'à 22h
"""  
Creuse_part_1 , Creuse_part_2, Plaine_part = 0 , 0, 0
for i in range(int(6/dt)):
    Creuse_part_1 += Pgrid_step[i] 
    
for i in range(int(22/dt),int(24/dt)):
   Creuse_part_2 += Pgrid_step[i] 
   
for i in range(int(6/dt),int(22/dt)):
    Plaine_part += Pgrid_step[i]
        
Pgrid_out_creuse = (Creuse_part_1+Creuse_part_2) / 1000                  # Division par 1000 pour mettre les valeurs en kWh 
Pgrid_out_plaine = Plaine_part / 1000                                    # Division par 1000 pour mettre les valeurs en kWh 
Pcons_unsatisfied_out=np.cumsum(Pcons_unsatisfied_step)/1000    # Division par 1000 pour mettre les valeurs en kWh 

Cout_achat = Cout_grid_Creuse * Pgrid_out_creuse + Cout_grid_plaine *Cout_grid_plaine
Cout_unsatisfied = - C_conso_unsatisfied * Pcons_unsatisfied_out[-1]
Ctotal = Cout_achat + Cout_unsatisfied
print(' Cout Achat de grid = {} \n Cout demande unsatisfied = {} \n Cout_Total = {}'.format(Cout_achat, Cout_unsatisfied, Ctotal) ,'Euros')
############################################################################################################
# %% Affichage

plt.figure(1)
plt.subplot(211)
plt.plot(Statofcharge)
plt.xlabel('Houre')
plt.ylabel('Energie')
plt.title('Bilan de Gestion Energie Par RL par Q_table obtenue' )
plt.legend((' SoC % '), loc='best', shadow=True)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Pnet1)
plt.plot(Pgrid_step)
plt.plot(Pprod_shed_step)
plt.plot(Pcons_unsatisfied_step)
plt.legend(('Pnet_last_day', 'Pgrid_last_day ', 'Pprod_shed_step_last_day', 'Pcons_unsatisfied_step_last_day'),
loc='best', shadow=True)
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.grid(True)
plt.show()


