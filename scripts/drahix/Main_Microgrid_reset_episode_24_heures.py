# -*- coding: utf-8 -*-

"""
Created on Fri May 15 10:30:13 2020

@author: mdini
"""
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from microgrid_ai.agent_microgrid import Agent
from microgrid_ai.microgrid_simulator import MicrogridSimulator

plt.close('all')

# Initialisation de nombre d'épisodes et la taille d'actions possible(Grid_On ou Grid_OFF):
n_episode = 10
n_actions = 2

DT = 0.5  # [h] Pas de discrétisation du temps
dp = 100  # Pas de discrétisation de Puissance
SoC_min, SoC_max = 0, 21000  # [W.h] Capacité min et max de la batterie                          

# %% Récupérer les données historiques de Pnet = Pload - Ppv  (Toutes les 30 minutes) :
"""
On prend une sample avec (le nombre de jour = n_episode) de notre data_base
Chaque épisode présente un jour choisi par hazard dans notre database

"""
month1 = pd.Series(pd.date_range('8/1/2016', freq='D', periods=31))
month2 = pd.Series(pd.date_range('8/1/2017', freq='D', periods=31))
database = pd.concat([month1, month2], axis=0, join='inner', ignore_index=True)

data_base = pd.DataFrame({'Dates': database})
data_base['Dates'] = data_base['Dates'].astype(str)

chosen_idx = np.random.choice(len(data_base), replace=True, size=n_episode)
sample_data_names = data_base.iloc[chosen_idx]

sample_data_names = 'DrahiX_5min_to_30Min@' + sample_data_names + '.txt'
sample_data_names = sample_data_names.set_index((np.arange(n_episode)))

data_path = "/home/admin/Documents/02-Recherche/02-Python/mohsen/microgrid_ai/data/drahix"

Training_data = pd.read_csv(os.path.join(data_path, str(sample_data_names.Dates[0])), delimiter="\t")
Training_data = Training_data.drop(Training_data.index[len(Training_data) - 1])

n_points = len(Training_data)

for i in range(1, n_episode):
    X = pd.read_csv(os.path.join(data_path, str(sample_data_names.Dates[i])), delimiter="\t")
    X = X.drop(X.index[len(X) - 1])
    Training_data = pd.concat([Training_data, X], axis=0, join='inner', ignore_index=True)

Pnet1 = ((Training_data.Cons - Training_data.Prod) // dp) * dp

# %%
t0 = time.time()
# %% Initialisation de l'environement et l'espace d'état:

Temp_ini = 0
Pnet_ini = Pnet1[0]
Pnet_min, Pnet_max = min(Pnet1), max(Pnet1)  # [W]
n_Time = int(round(24. / DT))
n_Pnet = int((Pnet_max - Pnet_min) / dp + 1)
n_SOC = int((SoC_max / dp) + 1)
n_state = n_SOC * n_Pnet * n_Time  # Initialisation de la taille de l'espace d'état

# SOC array
SoC = np.linspace(SoC_min, SoC_max, n_SOC)
# SOC_ini = 500                                    # Choisir l'état initiale manuele
# SOC_ini = 10500                                  # Choisir l'état initiale manuele
SOC_ini = int(np.random.choice(SoC))  # Choisir l'état initiale par hazard

# Pnet array
Pnet = np.linspace(Pnet_min, Pnet_max, n_Pnet)

# %% Initialisation de l'agent

# Fixer les coefficients de Rewards:
Cbatful, Cbat_empty, C_PV_shed, Cgrid_use_Creuse, Cgrid_use_plaine, Cbat_use = 0, 1000, 1000, 10, 100, 0

learning_rate = 0.9
discount = 0.95
exploration_rate = 1
iterations = n_episode * n_points

# %% Construire de l'environnement et de l'agent :
microgrid = MicrogridSimulator(n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC, n_Pnet, n_SOC, dp,
                               SoC_min, Pnet_min, Cbatful, C_PV_shed, Cbat_empty, Cgrid_use_Creuse,
                               Cgrid_use_plaine, Cbat_use, DT)

agent = Agent(n_state, n_episode, learning_rate, discount, exploration_rate, iterations,
              actions = microgrid.actions)

# %% main loop:

### Initialisation des paramètres ###
reward_episode = np.zeros(n_episode)

Pgrid_episode = np.zeros(n_episode)

Pgrid_step = np.zeros(n_episode * n_points)
Pprod_shed_step = np.zeros(n_episode * n_points)
Pcons_unsatisfied_step = np.zeros(n_episode * n_points)

last_total_reward = np.zeros(n_episode)

last_total_reward[0] = 0

Statofcharge = np.zeros(iterations)

Statofcharge_last_day = np.zeros(4 * n_points)
Pnet1_last_day = np.zeros(4 * n_points)
Pgrid_last_day = np.zeros(4 * n_points)
Pprod_shed_step_last_day = np.zeros(4 * n_points)
Pcons_unsatisfied_step_last_day = np.zeros(4 * n_points)

### Coeur de l'algorithme ###
for episode in range(n_episode):

    total_reward = 0
    total_Pgrid = 0

    for step in range(n_points):
        old_state = microgrid.index_state
        print('old_state= ', old_state)
        print('old_state= ', microgrid.get_soc_time_pnet())

        action = agent.get_next_action(old_state)  # Query agent for the next action
        print('action= ', action)

        new_state, reward, Pgrid, Pprod_shed, Pcons_unsatisfied = microgrid.take_action(action, Pnet1[
            (episode * n_points) + step])  # Take action, get new state and reward

        print(f'pnet demandé {Pnet1[(episode * n_points) + step]}')
        print(microgrid.get_soc_time_pnet())  # donne le nouveau soc,

        #        print('new_state = ',new_state,'reward  = ',reward )

        agent.update(old_state, new_state, action, reward)  # Let the agent update internals

        total_reward += reward

        total_Pgrid += Pgrid

        Statofcharge[(episode * n_points) + step] = microgrid.state_SOC[new_state]

        Pgrid_step[(episode * n_points) + step] = Pgrid

        Pprod_shed_step[(episode * n_points) + step] = Pprod_shed

        Pcons_unsatisfied_step[(episode * n_points) + step] = Pcons_unsatisfied

    Pgrid_episode[episode] = total_Pgrid

    reward_episode[episode] = total_reward

    last_total_reward[episode] = reward_episode[episode]

Final_Q_table = agent.q_table

# Calcule des différents variables au cours du temps:
for i in range(4 * n_points):
    Statofcharge_last_day[i] = Statofcharge[((n_episode - 4) * n_points) + i]
    Pnet1_last_day[i] = Pnet1[((n_episode - 4) * n_points) + i]
    Pgrid_last_day[i - 1] = Pgrid_step[((n_episode - 4) * n_points) + i]
    Pprod_shed_step_last_day[i - 1] = Pprod_shed_step[((n_episode - 4) * n_points) + i]
    Pcons_unsatisfied_step_last_day[i - 1] = Pcons_unsatisfied_step[((n_episode - 4) * n_points) + i]


# Calcule des différents variables au cours du temps:
# for i in range(n_points):
#    Statofcharge_last_day[i] = Statofcharge[((n_episode - 1) * n_points) + i]
#    Pnet1_last_day[i] = pnet[((n_episode - 1) * n_points) + i]
#    Pgrid_last_day[i-1] = Pgrid_step[((n_episode-1) * n_points) + i]
#    Pprod_shed_step_last_day[i-1] = Pprod_shed_step[((n_episode-1) * n_points) + i]
#    Pcons_unsatisfied_step_last_day[i-1] = Pcons_unsatisfied_step[((n_episode-1) * n_points) + i]       
# %% Calcule de Coût :
"""
Calculer le cout en considérant le tarif bleu 
Heures creuses = 22h jusqu'à 6h
Heures plaines = 6h jusqu'à 22h
"""
C_conso_unsatisfied = 10  # euros par KWh
Cout_grid_plaine = 0.133  # euros par KWh
Cout_grid_Creuse = 0.11  # euros par KWh

Creuse_part_1, Creuse_part_2, Plaine_part = 0, 0, 0

n_jour = int(len(Pgrid_last_day) / n_Time)

for j in range(n_jour):
    for i in range(int(6 / DT)):
        Creuse_part_1 += Pgrid_last_day[i + (j * n_Time)]

    for i in range(int(22 / DT), int(24 / DT)):
        Creuse_part_2 += Pgrid_last_day[i + (j * n_Time)]

    for i in range(int(6 / DT), int(22 / DT)):
        Plaine_part += Pgrid_last_day[i + (j * n_Time)]

Pgrid_out_creuse = (Creuse_part_1 + Creuse_part_2) / 1000  # Division par 1000 pour mettre les valeurs en kWh
Pgrid_out_plaine = Plaine_part / 1000  # Division par 1000 pour mettre les valeurs en kWh
Pcons_unsatisfied_out = np.cumsum(
    Pcons_unsatisfied_step_last_day) / 1000  # Division par 1000 pour mettre les valeurs en kWh

Cout_achat = Cout_grid_Creuse * Pgrid_out_creuse + Cout_grid_plaine * Cout_grid_plaine
Cout_unsatisfied = - C_conso_unsatisfied * Pcons_unsatisfied_out[-1]
Ctotal = Cout_achat + Cout_unsatisfied

print(
    ' Cout Achat de grid = {} \n Cout demande unsatisfied = {} \n Cout_Total = {}'.format(Cout_achat, Cout_unsatisfied,
                                                                                          Ctotal), 'Euros')
############################################################################################################
# %%
t_calcul = time.time() - t0
print(' Nombres pisodes = {} \n Temps de calcul = {}'.format(n_episode, t_calcul), 'seconds')
######################################################################################################
# %% Affichage

width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(2, 1)
axs[0].bar(x, Statofcharge_last_day, 2 * width, label='Statofcharge_last_day')
axs[1].bar(x - width, Pnet1_last_day, width, label='net_last_day')
axs[1].bar(x, Pgrid_last_day, width, label='Pgrid_last_day')
axs[1].bar(x + width, Pprod_shed_step_last_day, width, label='Pprod_shed_last_day')
axs[1].bar(x + (2 * width), Pcons_unsatisfied_step_last_day, width, label='Pcons_unsatisfied_last_day')
fig.suptitle('Bilan_Last_Day_RL')
axs[0].set_ylabel('Energie')
axs[0].set_xlabel('Houre')
axs[0].set_xticks(x)
# axs[0].set_xticklabels(labels)
axs[0].legend()
# axs[0].grid(True)
axs[1].set_ylabel('Puissance')
axs[1].set_xlabel('Houre')
axs[1].set_xticks(x)
# axs[1].set_xticklabels(labels)
axs[1].legend()
# axs[1].grid(True)

plt.figure(4)
plt.subplot(211)
plt.plot(Statofcharge_last_day)
plt.xlabel('Houre')
plt.ylabel('Energie')
plt.title('Bilan_Last_Day_RL')
plt.legend(('Statofcharge_last_day'), loc='best', shadow=True)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Pnet1_last_day)
plt.plot(Pgrid_last_day)
plt.plot(Pprod_shed_step_last_day)
plt.plot(Pcons_unsatisfied_step_last_day)
plt.legend(('Pnet_last_day', 'Pgrid_last_day ', 'Pprod_shed_step_last_day', 'Pcons_unsatisfied_step_last_day'),
           loc='best', shadow=True)
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.grid(True)
plt.show()
# %% Enregistrer le Q_table
#
# data_2write = pd.concat([Final_Q_table['GRID_OFF'],Final_Q_table['GRID_ON'] ],axis=1, join='inner', keys=['GRID_OFF','GRID_ON'],sort=False)
# file_write = 'Q_table' + '.txt'
# sep_write = '\t'
# pd.DataFrame.to_csv(data_2write,file_write,sep_write,index=True)

# %%
# Q_table_diff = Final_Q_table.GRID_OFF - Final_Q_table.GRID_ON
#
# Q_table_diff = pd.DataFrame((Q_table_diff), columns = ['Diff'])
# 
# Final_Q_table = pd.concat([Final_Q_table,Q_table_diff], axis = 1 )
#
#
# Table_ON = np.zeros([n_SOC+1, n_Pnet+1] )
# Table_OFF = np.zeros([n_SOC+1, n_Pnet+1] )
# Table_diff = np.zeros([n_SOC+1, n_Pnet+1] )

#################################################################################################        
# %%

# Table_ON = pd.DataFrame(0, index = SoC , columns = Pnet)
# Table_OFF = pd.DataFrame(0, index = SoC , columns = Pnet)
# Table_diff = pd.DataFrame(0, index = SoC , columns = Pnet)


#
# for i in range (1,n_SOC+1):
#    Table_ON[i][0]=(1000+dp)-dp*i
#    Table_OFF[i][0]=(1000+dp)-dp*i
#    Table_diff[i][0]=(1000+dp)-dp*i
# for i in range (1,n_Pnet+1):
#    Table_ON[0][i]=dp*i-(n_Pnet+1)*dp
#    Table_OFF[0][i]=dp*i-(n_Pnet+1)*dp
#    Table_diff[0][i]=dp*i-(n_Pnet+1)*dp
# for i in range (n_SOC,0,-1):
#    for j in range (1,n_Pnet+1):
#        Table_ON[i][j] = Final_Q_table[(np.abs(i-n_SOC))*n_Pnet+(j-1)][1]
#        Table_OFF[i][j] = Final_Q_table[(np.abs(i-n_SOC))*n_Pnet+(j-1)][0]
#        Table_diff[i][j] = Q_table_diff[(np.abs(i-n_SOC))*n_Pnet+(j-1)]
##print(Table_ON)
##print(Table_OFF)
##print(Table_diff)
#        
# data_ON = Table_ON
# data_OFF= Table_OFF
# data_diff = Table_diff

# plt.figure(2)
##sns.heatmap(data_ON, vmin=0, vmax=1 , annot=False , cmap="YlGnBu", linewidths=.3)
# sns.heatmap(Q_table_diff, annot=False , cmap="YlGnBu", linewidths=.0)
# plt.title('Grid_ON')
# plt.show()

# plt.figure(3)
# sns.heatmap(data_OFF,vmin=0, vmax=1 , annot=True , cmap="YlGnBu", linewidths=.3)
# plt.title('Grid_OFF')
# plt.show()
#
# plt.figure(4)
# sns.heatmap(data_diff , annot=True , cmap="YlGnBu", linewidths=.3)
# plt.title('Diff_ON_OFF')
# plt.show()

# plt.figure(5)
# plt.plot(Statofcharge,'bo-')
# plt.xlabel('Steps(hours)= episode*length database')
# plt.ylabel('SoC')
# plt.title('SoC_Pendant_training')
# plt.grid(True)

# plt.figure(6)
# plt.plot(reward_episode,'yo-')
# plt.xlabel('episods')
# plt.ylabel('cumulative rewards per episode')
# plt.title('Reward_Par_episode')
# plt.show()
# plt.grid(True)

# plt.figure(7)
# plt.plot(performance,'bo-')
# plt.xlabel('episods')
# plt.ylabel('Actuel reward  - Last reward')
# plt.title('Difference reward between two episodes')
# plt.show()
# plt.grid(True)

# plt.figure(8)
# plt.plot(Pgrid_episode,'go-')
# plt.plot(Pgrid_episode[-1]*10,'ro-')
# plt.xlabel('steps')
# plt.ylabel('Pgrid')
# plt.title('Pgrid_Par_episode')
# plt.show()
# plt.grid(True)

# plt.figure(9)
# plt.plot(pnet,'ro-')
# plt.xlabel('Houre')
# plt.ylabel('Puissance')
# plt.title('Pnet_chaque_instant')
# plt.show()
# plt.grid(True)
