# -*- coding: utf-8 -*-

"""
Created on Fri May 15 10:30:13 2020

@author: mdini
"""
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from agent_microgrid import Agent
from microgrid_simulator import MicrogridSimulator
plt.close('all')

#%% Récuperer les données historiques de Pnet = Pload - Ppv  (Toutes les 30 minutes)
file1 = 'DrahiX_5min_to_30Min@2018-08-01' 
file2 = 'DrahiX_5min_to_30Min@2018-08-02'
file3 = 'DrahiX_5min_to_30Min@2018-08-03'
file4 = 'DrahiX_5min_to_30Min@2018-08-04'
file5 = 'DrahiX_5min_to_30Min@2018-08-05'
file6 = 'DrahiX_5min_to_30Min@2018-08-06'
file7 = 'DrahiX_5min_to_30Min@2018-08-07' 

file_read_1 = file1 + '.txt'
file_read_2 = file2 + '.txt'
file_read_3 = file3 + '.txt'
file_read_4 = file4 + '.txt'
file_read_5 = file5 + '.txt'
file_read_6 = file6 + '.txt'
file_read_7 = file7 + '.txt'

Informations_1 = pd.read_csv(file_read_1,delimiter = "\t")
Informations_2 = pd.read_csv(file_read_2,delimiter = "\t")
Informations_3 = pd.read_csv(file_read_3,delimiter = "\t")
Informations_4 = pd.read_csv(file_read_4,delimiter = "\t")
Informations_5 = pd.read_csv(file_read_5,delimiter = "\t")
Informations_6 = pd.read_csv(file_read_6,delimiter = "\t")
Informations_7 = pd.read_csv(file_read_7,delimiter = "\t")

Pnet_1 = ((Informations_1.Cons-Informations_1.Prod)//10)*10
Pnet_1 = Pnet_1.drop(Pnet_1.index[len(Pnet_1)-1])

Pnet_2 = ((Informations_2.Cons-Informations_2.Prod)//10)*10
Pnet_2 = Pnet_2.drop(Pnet_2.index[len(Pnet_2)-1])

Pnet_3 = ((Informations_3.Cons-Informations_3.Prod)//10)*10
Pnet_3 = Pnet_3.drop(Pnet_3.index[len(Pnet_3)-1])

Pnet_4 = ((Informations_4.Cons-Informations_4.Prod)//10)*10
Pnet_4 = Pnet_4.drop(Pnet_4.index[len(Pnet_4)-1])

Pnet_5 = ((Informations_5.Cons-Informations_5.Prod)//10)*10
Pnet_5 = Pnet_5.drop(Pnet_5.index[len(Pnet_5)-1])

Pnet_6 = ((Informations_6.Cons-Informations_6.Prod)//10)*10
Pnet_6 = Pnet_6.drop(Pnet_6.index[len(Pnet_6)-1])

Pnet_7 = ((Informations_7.Cons-Informations_7.Prod)//10)*10
Pnet_7 = Pnet_7 .drop(Pnet_7 .index[len(Pnet_7 )-1])

Pnet1 = pd.concat( [Pnet_1,Pnet_2,Pnet_3,Pnet_4,Pnet_5,Pnet_6,Pnet_7] , axis=0, join='inner',ignore_index=True)
hamer1 = np.where( Pnet1 >=  5000 )
Pnet1.loc[(hamer1)] = 5000
#hamer2 = np.where( Pnet1 <= - 8000 )
#Pnet1.loc[(hamer2)] = -8000
################################################################################################################### 
#%% Récuperer les données historiques de Pnet = Pload - Ppv 
#with open("Profil_4.csv", newline='') as csvfile :
#    reader = csv.reader(csvfile)
#    liste = []
#    for row in reader:
#        liste += row
#n_points = len(liste)
#Pnet1 = np.array([0.]*n_points)
#for r in range(n_points) :
#    Pnet1[r] = float(liste[r])
#Pnet1 = pd.Series(Pnet1)
####################################################################################################################  
#%%
t0 = time.time()
#%% Initialisation de nombre d'épisodes et la taille d'actions possible(Grid_On ou Grid_OFF):
n_actions= 2                                
n_episode= 10000                             

#Initialisation de l'environement et l'espace d'état:
n_points = len(Pnet1)

Temp_ini = 0
Pnet_ini = Pnet1[0]

#SOC_ini = 500 # Choisir l'état initiale
#DT = 1      #[h]   
#dp = 100    # Pas de descritisatio
#SoC_min, SoC_max = 0, 1000    # [W.h]

SOC_ini = 10500  # Choisir l'état initiale
DT = 0.5     #[h]                                                                  
dp = 10      # Pas de descritisation                                  
SoC_min, SoC_max = 0, 21000         # [W.h]
                

Pnet_min, Pnet_max = min(Pnet1),max(Pnet1)  # [W]

n_Time = int(round(24./DT))

n_Pnet = int((Pnet_max - Pnet_min)/dp+1)

n_SOC = int((SoC_max/dp)+1)

n_state = n_SOC*n_Pnet*n_Time                     # Initialisation de la taille de l'espace d'état 
 
# SOC array
SoC = np.linspace(SoC_min,SoC_max,n_SOC)

# Pnet array
Pnet = np.linspace(Pnet_min,Pnet_max,n_Pnet)

# Fixer les coefficients de Rewards:
Cbatful, Cbat_empty,  C_PV_shed , Cgrid_use_Creuse, Cgrid_use_plaine , Cbat_use = 0, 1000, 1000, 10, 100, 0
 
#### Initialisation et Choisir agent parmi(Agent(Q-learning), Accountant, Drunkard)
learning_rate = 0.9  
discount = 0.95
exploration_rate = 1 
iterations = n_episode*n_points
###############################################################################################################
#%% Construire de l'environement et l'agent:
microgrid = MicrogridSimulator(n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC,n_Pnet,n_SOC, dp,
                              SoC_min, Pnet_min, DT, Cbatful, C_PV_shed, Cbat_empty, Cgrid_use_Creuse, Cgrid_use_plaine, Cbat_use)

agent=Agent( n_state, n_episode, n_actions, learning_rate, discount, exploration_rate , iterations)
################################################################################################################  
#%% main loop:    

### Initialisation des paramètres ###
reward_episode = np.zeros(n_episode)

Pgrid_episode = np.zeros(n_episode)

Pgrid_step = np.zeros(n_episode*n_points)
Pprod_shed_step=np.zeros(n_episode*n_points)
Pcons_unsatisfied_step=np.zeros(n_episode*n_points)

last_total_reward=np.zeros(n_episode)

last_total_reward[0] = 0 

Statofcharge=np.zeros(iterations)

Statofcharge_last_day = np.zeros(n_points)
Pnet1_last_day = np.zeros(n_points)
Pgrid_last_day = np.zeros(n_points)
Pprod_shed_step_last_day= np.zeros(n_points)
Pcons_unsatisfied_step_last_day= np.zeros(n_points)
    

### Coeur de l'algorithme ###
for episode in range (n_episode):
    microgrid.reset()                                                 # Remettre l'agent dans son état initiale après chaque épisode
    total_reward=0
    total_Pgrid=0
                                     
    for step in range(n_points):
        
        old_state = microgrid.index_state
#        print('old_state= ',old_state)

        action = agent.get_next_action(old_state)                     # Query agent for the next action
#        print('action= ',action)

        new_state, reward ,Pgrid, Pprod_shed, Pcons_unsatisfied = microgrid.take_action(action,Pnet1[step]) # Take action, get new state and reward
#        print('new_state = ',new_state,'reward  = ',reward )
        
        agent.update(old_state, new_state, action, reward)            # Let the agent update internals
        
        total_reward += reward  
        total_Pgrid += Pgrid
        
        Statofcharge[(episode*n_points)+step] = microgrid.state_SOC[new_state] 
        
        Pgrid_step[(episode*n_points)+step] = Pgrid
        
        Pprod_shed_step[(episode*n_points)+step] = Pprod_shed
        
        Pcons_unsatisfied_step[(episode*n_points)+step] = Pcons_unsatisfied

        
    Pgrid_episode[episode] = total_Pgrid
    
    reward_episode[episode] = total_reward
    
    last_total_reward[episode] = reward_episode[episode]
    
#print("Final Q-table", agent.q_table )
    
############################################################################## 
#%% Calcule des différents variables au cours du temps:   
for i in range(n_points):
    Statofcharge_last_day[i] = Statofcharge[((n_episode-1)*n_points)+i]
    Pnet1_last_day[i] = Pnet1[i]
    Pgrid_last_day[i-1]=Pgrid_step[((n_episode-1)*n_points)+i]
    Pprod_shed_step_last_day[i-1]=Pprod_shed_step[((n_episode-1)*n_points)+i]
    Pcons_unsatisfied_step_last_day[i-1]=Pcons_unsatisfied_step[((n_episode-1)*n_points)+i]
    
#################################################################################################        
#%% Calcule de Coût :
"""
Calculer le cout en considérant le tarif bleu 
Heures creuses = 22h jusqu'à 6h
Heures plaines = 6h jusqu'à 22h
"""  
C_conso_unsatisfied= 10   #euros par KWh
Cout_grid_plaine = 0.133  #euros par KWh
Cout_grid_Creuse = 0.11   #euros par KWh

Creuse_part_1 , Creuse_part_2, Plaine_part = 0 , 0, 0
n_jour= int (len(Pgrid_last_day)/n_Time)

for j in range (n_jour):
    for i in range(int(6/DT)):
        Creuse_part_1 += Pgrid_last_day[i+(j*n_Time)]
        
    for i in range(int(22/DT),int(24/DT)):
       Creuse_part_2 += Pgrid_last_day[i+(j*n_Time)] 
       
    for i in range(int(6/DT),int(22/DT)):
        Plaine_part += Pgrid_last_day[i+(j*n_Time)]
        
Pgrid_out_creuse = (Creuse_part_1+Creuse_part_2) / 1000                  # Division par 1000 pour mettre les valeurs en kWh 
Pgrid_out_plaine = Plaine_part / 1000                                    # Division par 1000 pour mettre les valeurs en kWh 
Pcons_unsatisfied_out=np.cumsum(Pcons_unsatisfied_step_last_day)/1000    # Division par 1000 pour mettre les valeurs en kWh 

Cout_achat = Cout_grid_Creuse * Pgrid_out_creuse + Cout_grid_plaine *Cout_grid_plaine
Cout_unsatisfied = - C_conso_unsatisfied * Pcons_unsatisfied_out[-1]
Ctotal = Cout_achat + Cout_unsatisfied


print(' Cout Achat de grid = {} \n Cout demande unsatisfied = {} \n Cout_Total = {}'.format(Cout_achat, Cout_unsatisfied, Ctotal) ,'Euros')
############################################################################################################
#%%
time.time()-t0
print('Temps de calcul = {}'.format (time.time()-t0) ,'seconds')
######################################################################################################
#%% Affichage

width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(2, 1)
axs[0].bar(x             ,Statofcharge_last_day          ,2*width, label='Statofcharge_last_day' )
axs[1].bar(x  - width    ,Pnet1_last_day                 , width  , label='net_last_day')
axs[1].bar(x             ,Pgrid_last_day                 , width  , label='Pgrid_last_day') 
axs[1].bar(x + width     ,Pprod_shed_step_last_day       , width  , label='Pprod_shed_last_day')
axs[1].bar(x + (2*width) ,Pcons_unsatisfied_step_last_day, width  , label='Pcons_unsatisfied_last_day')
fig.suptitle('Bilan_Last_Day_RL')
axs[0].set_ylabel('Energie')
axs[0].set_xlabel('Houre')
axs[0].set_xticks(x)
#axs[0].set_xticklabels(labels)
axs[0].legend()
#axs[0].grid(True)
axs[1].set_ylabel('Puissance')
axs[1].set_xlabel('Houre')
axs[1].set_xticks(x)
#axs[1].set_xticklabels(labels)
axs[1].legend()
#axs[1].grid(True)