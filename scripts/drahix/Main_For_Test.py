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

# %% Initialization of environment and state space:
first_day_of_database = '8/1/2016'
start_day_of_month_test = 1
n_jour_test = 7

# le tarif bleu du réseau :
C_conso_unsatisfied = 10  # euros par KWh
Cout_grid_plaine = 0.133  # euros par KWh
Cout_grid_Creuse = 0.11  # euros par KWh

SOC_ini = 10500                                  # Choisir l'état initiale manuele
#SOC_ini =int(np.random.choice(SoC))             # Choisir l'état initiale par hazard
 
#%% Récuperer la Q_Table obtenue par les données historiques
#This information is available from the training phase:
data_path = r"C:\Users\mdini\Documents\GitHub\microgrid_ai\data\drahix"
file = 'Q_table_Predict_2000 episodes_20 percent incertity'
info = 'information_Q_table_Predict_2000 episodes_20 percent incertity'
file_read = file + '.txt'
info_read = info + '.txt'
q_table = pd.read_csv(os.path.join(data_path, file_read), delimiter="\t")
infos = pd.read_csv(os.path.join(data_path, info_read), delimiter="\t")
dp = infos.loc[0]['dp']   # Pas de discrétisation de Puissance à partir de pas obtenu sur la phase Training
dt = infos.loc[0]['dt']   # [h] Pas de discrétisation du temps
Pnet_min_q_table = infos.loc[0]['Pnet_min_q_table']  # [W]
Pnet_max_q_table = infos.loc[0]['Pnet_max_q_table']  # [W]
SoC_min = infos.loc[0]['SoC_min'] # [W.h] Capacité min de la batterie 
SoC_max = infos.loc[0]['SoC_max'] # [W.h] Capacité max de la batterie 

# %% Récupérer les données historiques de Pnet = Pload - Ppv  (Toutes les 30 minutes) :
"""
On prend une sample avec (le nombre de jour = n_episode) de notre data_base
Chaque épisode présente un jour choisi par hazard dans notre database

"""
month = pd.Series(pd.date_range(first_day_of_database, freq='D', periods=31))

data_base = pd.DataFrame({'Dates': month})
data_base['Dates'] = data_base['Dates'].astype(str)

#chosen_idx = np.random.choice(len(data_base), replace=True, size = n_jour_test)           # Choose days randomely
chosen_idx = np.arange(start_day_of_month_test -1 , start_day_of_month_test + n_jour_test-1)     # Choose days by orders

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

#%% Initialisation de l'environement et l'espace d'état:
n_points = len(Pnet1)
n_jours = int( n_points / (24 / dt) )
Temp_ini = 0
Pnet_ini = Pnet1[0]
Pnet_min, Pnet_max = min(Pnet1),max(Pnet1)  # [W]
n_Time = int(round(24./dt))
n_Pnet = int((Pnet_max - Pnet_min)/dp+1)
n_SOC = int((SoC_max/dp)+1)
n_state = n_SOC*n_Pnet*n_Time                     # Initialisation de la taille de l'espace d'état
SoC = np.linspace(SoC_min,SoC_max,n_SOC)          # SOC array
Pnet = np.linspace(Pnet_min,Pnet_max,n_Pnet)      # Pnet array

#%% Test of compatibility of Q_table
if (Pnet_min_q_table > Pnet_min):
    add_head = (abs(Pnet_min_q_table - Pnet_min))/dp
    add_head_q_table = add_head*n_SOC*n_Time
    head_part= pd.DataFrame(0, index=np.arange(add_head_q_table), columns={'Unnamed: 0','GRID_OFF','GRID_ON'})
    q_table = pd.concat([head_part, q_table], axis=0, join='inner', ignore_index=True)
    
elif(Pnet_max_q_table < Pnet_max):
    add_fin = (abs(Pnet_max_q_table - Pnet_max))/dp
    add_fin_q_table = add_fin*n_SOC*n_Time
    fin_part = pd.DataFrame(0, index=np.arange(add_fin_q_table), columns={'Unnamed: 0','GRID_OFF','GRID_ON'})
    q_table = pd.concat([q_table,fin_part], axis=0, join='inner', ignore_index=True)


#%% Construire de l'environement et l'agent:
microgrid = MicrogridSimulator(n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC,n_Pnet,n_SOC, dp, SoC_min, Pnet_min, dt)

agent = Agent(q_table)

#%% main loop:    
t0 = time.time()
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
    
    Statofcharge[step] = microgrid.state_SOC.loc[new_state,'state_SOC']
    
    Pgrid_step[step-1] = Pgrid
    
    Pprod_shed_step[step-1] = Pprod_shed
    
    Pcons_unsatisfied_step[step-1] = Pcons_unsatisfied
        
# %% Calcule de Coût :
"""
Calculer le cout en considérant le tarif bleu 
Heures creuses = 22h jusqu'à 6h
Heures plaines = 6h jusqu'à 22h
"""  
Creuse_part_1 , Creuse_part_2, Plaine_part = 0 , 0, 0

for i in range (n_jours):
    
    for j in range  (int(6/dt)):
        Creuse_part_1 += Pgrid_step[ i*int(24/dt) + j]    / 1000    # Division par 1000 pour mettre les valeurs en kWh 
    
    for j in range(int(6/dt),int(22/dt)):
        Plaine_part += Pgrid_step[ i*int(24/dt) + j]      / 1000    # Division par 1000 pour mettre les valeurs en kWh 
        
    for j in range(int(22/dt),int(24/dt)):
        Creuse_part_2 += Pgrid_step[ i *int (24/dt) + j]  / 1000    # Division par 1000 pour mettre les valeurs en kWh 
       
Pgrid_out_creuse = (Creuse_part_1+Creuse_part_2)                   
Pgrid_out_plaine = Plaine_part                                     
Pcons_unsatisfied_out=np.cumsum(Pcons_unsatisfied_step)   / 1000    # Division par 1000 pour mettre les valeurs en kWh 

Cout_achat = ( Cout_grid_Creuse * Pgrid_out_creuse * dt ) + ( Cout_grid_plaine * Cout_grid_plaine * dt )
Cout_unsatisfied = - C_conso_unsatisfied * Pcons_unsatisfied_out[-1] * dt
Ctotal = Cout_achat + Cout_unsatisfied
print(' Grid purchase cost = {:8.2f} \n Cost of unsatisfied demand = {:8.2f} \n Total Cout = {:8.2f}'.format(Cout_achat, Cout_unsatisfied, Ctotal) ,'Euros')

#%% Préparation de xticks:
freq_heur_affich = 6
step_xtick= 2*freq_heur_affich # Il faut mettre une valeure deux fois plus que freq_affich
freq_affich = str(freq_heur_affich)+'H'
rng = pd.Series(pd.date_range(first_day_of_database, freq=freq_affich, periods = n_jour_test*4))
rng=pd.DataFrame({'rng': rng})
rng=rng['rng'].astype(str)
out = [x[11:-3] for x in rng]

# %% Affichage
plt.figure(1)
plt.subplot(211)
plt.plot(Pnet1)
plt.plot(Pgrid_step)
plt.plot(Pprod_shed_step)
plt.plot(Pcons_unsatisfied_step)
plt.title('Energy Management Report for RL by applying Q_table obtained by Training Phase')
plt.legend(('Pnet', 'Pgrid', 'Pprod_shed', 'Pcons_unsatisfied'), loc='best', shadow=True)
plt.xlabel('Hour')
plt.ylabel('Power (W)')
plt.xticks(ticks = np.arange(0,len(Statofcharge),12), labels = out)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Statofcharge)
plt.xlabel('Hour')
plt.ylabel('Energy (Wh)')
plt.xticks(ticks = np.arange(0,len(Statofcharge),step_xtick), labels = out)
plt.legend(('SOC (\%)'), loc='best')
plt.grid(True)
plt.show()


