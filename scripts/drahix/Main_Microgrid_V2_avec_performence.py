# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:55:12 2020

@author: mdini
"""
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
from microgrid_ai.agent_microgrid import Agent
from microgrid_ai.microgrid_simulator import MicrogridSimulator
plt.close('all')

# %% Collect historical data from Pnet = Pload - Ppv (Every 30 minutes):
"""
We take a sample with (the number of days = n_episode) from our data_base
Each episode presents a day chosen by chance in our database
"""
def database(months_of_train, months_of_test, nombr_jour_Train, nombr_jour_Test):
    
    data_path = r"C:\Users\mdini\Documents\GitHub\microgrid_ai\data\drahix"
    # Database initialization:
    day_database_train = []
    database_train = [] 
    day_database_test = []
    database_test = []
 
    """
    In this part i shold to fix the problem to put more than 2 months for database
    (the problem is about ythe use of pd.concat)!!!!!
    """
    #Create the training database:
    for i in range (len(months_of_train)):         
        day_database_train.append(months_of_train[i])      
        database_train.append(pd.Series(pd.date_range(day_database_train[i], freq='D', periods=31))) 
#    print('r=',  database_train,'size=',np.shape(database_train))
    for i in range (len(months_of_train)-1):   
        database_train = pd.concat([database_train[i], database_train[i+1]], axis=0, join='inner', ignore_index=True)
#    print(database_train)
    data_base_train = pd.DataFrame({'Dates': database_train})
    data_base_train['Dates'] = data_base_train['Dates'].astype(str)
    chosen_idx_train = np.random.choice(len(data_base_train), replace=True, size=nombr_jour_Train)
    sample_data_names_train = data_base_train.iloc[chosen_idx_train] 
    sample_data_names_train = 'DrahiX_5min_to_30Min@' + sample_data_names_train + '.txt'
    sample_data_names_train = sample_data_names_train.set_index((np.arange(nombr_jour_Train)))   
    Training_data = pd.read_csv(os.path.join(data_path, str(sample_data_names_train.Dates[0])), delimiter="\t")
    Training_data = Training_data.drop(Training_data.index[len(Training_data) - 1])   
    n_points = len(Training_data)  
    for i in range(1, nombr_jour_Train):
        X = pd.read_csv(os.path.join(data_path, str(sample_data_names_train.Dates[i])), delimiter="\t")
        X = X.drop(X.index[len(X) - 1])
        Training_data = pd.concat([Training_data, X], axis=0, join='inner', ignore_index=True)  


    # Create the testing database:
    for i in range (len(months_of_test)):         
        day_database_test.append(months_of_test[i])      
        database_test.append(pd.Series(pd.date_range(day_database_test[i], freq='D', periods=31)))        
    for i in range (len(months_of_test)-1):   
        database_test = pd.concat([database_test[i], database_test[i+1]], axis=0, join='inner', ignore_index=True)
    
    data_base_test = pd.DataFrame({'Dates': database_test})
    data_base_test['Dates'] = data_base_test['Dates'].astype(str)
    chosen_idx_test = np.random.choice(len(data_base_test), replace=True, size=nombr_jour_Test)
    sample_data_names_test = data_base_test.iloc[chosen_idx_test]       
    sample_data_names_test = 'DrahiX_5min_to_30Min@' + sample_data_names_test + '.txt'
    sample_data_names_test = sample_data_names_test.set_index((np.arange(nombr_jour_Test)))   
    Testini_data = pd.read_csv(os.path.join(data_path, str(sample_data_names_test.Dates[0])), delimiter="\t")
    Testini_data = Testini_data.drop(Testini_data.index[len(Testini_data) - 1])   
    n_points_test = len(Testini_data)         
    for i in range(1, nombr_jour_Test):
        X = pd.read_csv(os.path.join(data_path, str(sample_data_names_test.Dates[i])), delimiter="\t")
        X = X.drop(X.index[len(X) - 1])
        Testini_data = pd.concat([Testini_data, X], axis=0, join='inner', ignore_index=True) 
        
        
    Pnet1_brut = (Training_data.Cons - Training_data.Prod)   
    Pnet_min_brut, Pnet_max_brut = min(Pnet1_brut), max(Pnet1_brut)  # [W]    
    dp = np.round( percent_pas * (Pnet_max_brut - Pnet_min_brut))// 10 * 10   # Pas de discrétisation de Puissance
    
    Pnet1_train = ((Training_data.Cons - Training_data.Prod) // dp) * dp
    Pnet1_test = ((Testini_data.Cons - Testini_data.Prod) // dp) * dp
    
    return n_points_test, n_points, Pnet1_train, Pnet1_test, dp

#%%
def split_list(alist, wanted_parts):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

#%% 
def algo (teau_incertitude, dt, Cout_grid_Creuse, Cout_grid_plaine, Cout_conso_unsatisfied, cout_achat_episode, cout_insatisfait_episode, total_reward, total_Pgrid, totale_cout, Pnet1, indicat, episode, n_points):

    for step in range(n_points):
            
        old_state = microgrid.index_state
            
        action = agent.get_next_action(old_state,indicat)
            
        new_state, reward, P_reel, Pgrid, Pprod_shed, Pcons_unsatisfied = microgrid.take_action(teau_incertitude, action, Pnet1[(episode * n_points) + step])  # Take action, get new state and reward
        
        agent.update(old_state, new_state, action, reward, indicat)  # Let the agent update internals
        
        
        if (int(6/dt) <= step < int(22/dt)):
            cout_achat = Cout_grid_plaine * (Pgrid/1000) * dt
        else:
            cout_achat = Cout_grid_Creuse * (Pgrid/1000) * dt
            
        cout_insatisfait = - Cout_conso_unsatisfied * (Pcons_unsatisfied/1000) * dt
        
        total_reward += reward
            
        total_Pgrid += Pgrid
        
        cout_achat_episode += cout_achat 
        
        cout_insatisfait_episode += cout_insatisfait 
        
        totale_cout += (cout_achat_episode + cout_insatisfait_episode) 
        
        Statofcharge[(episode * n_points) + step] = microgrid.env.loc[new_state,'state_SOC']
            
        Pgrid_step[(episode * n_points) + step] = Pgrid
        
        P_reel_step[(episode * n_points) + step] = P_reel
            
        Pprod_shed_step[(episode * n_points) + step] = Pprod_shed
            
        Pcons_unsatisfied_step[(episode * n_points) + step] = Pcons_unsatisfied  
    
         
    reward_episode[episode] = np.abs(total_reward)
    Pgrid_episode[episode] = total_Pgrid 
    totale_cout_achat [episode] = cout_achat_episode
    totale_cout_insatisfait [episode] = cout_insatisfait_episode
    cout_episode[episode] = totale_cout 
        
    if(indicat == 'Test'):
        
       performence.append(cout_episode[episode])
            
    return reward_episode, Pgrid_episode, performence
 
# %% Initialization of the System:
"""
"""
n_episode = 100
teau_incertitude = 0  #It must be in percentage
nombr_jour_Train = n_episode
n_jour_visio = 7
delta_episode_performence = int(n_episode/10)
nombr_jour_boxplot = 1
nombr_jour_Test =  nombr_jour_boxplot * int( n_episode / delta_episode_performence )
dt = 0.5  # [h] Step discretization of time
percent_pas = 0.05 # Percentage for the step discretization of Power
SoC_min, SoC_max = 0, 21000  # [W.h] Min and max battery capacity
#Temp_affich_heatmap = 18.0 # in hour in 30 minute steps
# the blue tariff of network :
Cout_conso_unsatisfied = 10  # euros per KWh
Cout_grid_plaine = 0.133  # euros per KWh
Cout_grid_Creuse = 0.110  # euros per KWh

months_of_train = ['10/1/2016','10/1/2017']
months_of_test =  ['10/1/2016','10/1/2017']
n_points_test, n_points, Pnet1_train, Pnet1_test, dp = database(months_of_train, months_of_test, nombr_jour_Train, nombr_jour_Test)
  
# %%
t0 = time.time()                       

# %% Initialization of environment and state space:
Temp_ini = 0
Pnet_ini = Pnet1_train[0]
Pnet_min, Pnet_max = min(Pnet1_train), max(Pnet1_train)  # [W]

n_Time = int(round(24. / dt))
n_Pnet = int((Pnet_max - Pnet_min) / dp + 1)
n_SOC = int((SoC_max / dp) + 1)
n_state = n_SOC * n_Pnet * n_Time  # the size of the state space
# SOC array
SoC = (np.linspace(SoC_min, SoC_max, n_SOC)// dp) * dp
# SOC_ini = 500                                    # Choose initial state manualy 
# SOC_ini = 10500                                  # Choose initial state manualy 
SOC_ini = int(np.random.choice(SoC))               # Choose the initial state randomely
# Pnet array
Pnet = np.linspace(Pnet_min, Pnet_max, n_Pnet)

# Agent initialization :
learning_rate    = 0.9
discount         = 0.95
exploration_rate = 1
iterations = n_episode * n_points

# Set the Rewards coefficients
Cbatful, Cbat_empty, C_PV_shed, Cgrid_use_Creuse, Cgrid_use_plaine, Cbat_use = 0, 1000, 1000, 10, 100, 10

# %% Build the environment and the agent :
microgrid = MicrogridSimulator(n_Time, Pnet_ini, SOC_ini, Temp_ini, Pnet, SoC, n_Pnet, n_SOC, dp,
                               SoC_min, Pnet_min, Cbatful, C_PV_shed, Cbat_empty, Cgrid_use_Creuse,
                               Cgrid_use_plaine, Cbat_use, dt)

agent = Agent(n_state, n_episode, learning_rate, discount, exploration_rate, iterations,
              actions = microgrid.actions)

# %% main loop:
# Initialization of parameters ###
reward_episode                  = np.zeros(n_episode)
Pgrid_episode                   = np.zeros(n_episode)
cout_episode                    = np.zeros(n_episode)
totale_cout_achat               = np.zeros(n_episode)
totale_cout_insatisfait         = np.zeros(n_episode)
Pgrid_step                      = np.zeros(n_episode * n_points)
P_reel_step                     = np.zeros(n_episode * n_points)
Pprod_shed_step                 = np.zeros(n_episode * n_points)
Pcons_unsatisfied_step          = np.zeros(n_episode * n_points)
Statofcharge                    = np.zeros(iterations)
performence                     = []
Statofcharge_last_day           = np.zeros(n_jour_visio * n_points)
Pnet1_last_day                  = np.zeros(n_jour_visio * n_points)
Pgrid_last_day                  = np.zeros(n_jour_visio * n_points)
P_reel_last_day                 = np.zeros(n_jour_visio * n_points)
Pprod_shed_step_last_day        = np.zeros(n_jour_visio * n_points)
Pcons_unsatisfied_step_last_day = np.zeros(n_jour_visio * n_points)

for episode in range(n_episode):
    
    total_reward = 0
    total_Pgrid  = 0
    totale_cout  = 0
    cout_achat_episode = 0
    cout_insatisfait_episode = 0

    if (episode % delta_episode_performence < nombr_jour_boxplot):
        episode_test = (episode//delta_episode_performence) + (episode % delta_episode_performence)
        indicat = 'Test'
        Pnet1 = Pnet1_test
        reward_episode_test, Pgrid_episode_test, performence = algo (teau_incertitude, dt, Cout_grid_Creuse, Cout_grid_plaine, Cout_conso_unsatisfied, cout_achat_episode, cout_insatisfait_episode, total_reward, total_Pgrid, totale_cout, Pnet1, indicat, episode_test, n_points )
        
    indicat = 'Train'
    Pnet1 = Pnet1_train   
    reward_episode, Pgrid_episode, performence = algo (teau_incertitude, dt, Cout_grid_Creuse, Cout_grid_plaine, Cout_conso_unsatisfied, cout_achat_episode, cout_insatisfait_episode, total_reward, total_Pgrid, totale_cout, Pnet1, indicat, episode, n_points) 
       
Final_Q_table = agent.q_table

#%% calculate calculation time:
t_calcul = time.time() - t0
print(' Nombre of episodes = {} \n dp = {} \n Pnet_min = {} \n Pnet_max = {} \n Calculation time = {:8.2f}'.format(n_episode,dp, Pnet_min, Pnet_max, t_calcul ), 'seconds')
# %% Postprocessing:  

#%% Calculates different variables over time:

for i in range(n_jour_visio * n_points):
    
    Statofcharge_last_day[i]               = Statofcharge[((n_episode - n_jour_visio) * n_points) + i]
    
    Pnet1_last_day[i]                      = Pnet1[((n_episode - n_jour_visio) * n_points) + i]
    
    P_reel_last_day[i-1]                   = P_reel_step[((n_episode - n_jour_visio) * n_points) + i]
    
    Pgrid_last_day[i - 1]                  = Pgrid_step[((n_episode - n_jour_visio) * n_points) + i]
    
    Pprod_shed_step_last_day[i - 1]        = Pprod_shed_step[((n_episode - n_jour_visio) * n_points) + i]
    
    Pcons_unsatisfied_step_last_day[i - 1] = Pcons_unsatisfied_step[((n_episode - n_jour_visio) * n_points) + i]

# %%Cost Calculation :
"""
Calculate the cost by considering the blue tariff for the number of days displayed
Off-peak hours = 10 p.m. until 6 a.m.
Peak hours = 6 a.m. until 10 p.m.
"""
Cout_achat = 0
Cout_unsatisfied = 0
Ctotal = 0
for i in range(n_episode - n_jour_visio , n_episode ):  
    Cout_achat += totale_cout_achat[i]
    Cout_unsatisfied += totale_cout_insatisfait[i]
Ctotal = Cout_achat + Cout_unsatisfied    
print( ' Grid purchase cost = {:8.2f} [euros] for {} last days \n Cost of unsatisfied demand = {:8.2f} [euros] for {} last days \n Total Cout = {:8.2f} [euros] pour {} derniers jours'.format(Cout_achat,str(n_jour_visio), Cout_unsatisfied,str(n_jour_visio), Ctotal,str(n_jour_visio)))    
# %% Visualize the Heatmap of Q_Table:
#
### The difference between two actions:
#heat_q_table = Final_Q_table.copy()
#
##check the connection situation:
#for i in range (n_state):
#    if(heat_q_table.loc [i,'GRID_OFF'] < heat_q_table.loc [i,'GRID_ON']): 
#        heat_q_table.loc [i,'GRID_OFF'] = 0
#        heat_q_table.loc [i,'GRID_ON'] = 1
#    else:
#        heat_q_table.loc [i,'GRID_OFF'] = 1
#        heat_q_table.loc [i,'GRID_ON'] = 0   
#        
#q_table_diff = pd.DataFrame({'q_table_diff': (Final_Q_table.loc [:,'GRID_OFF'] - Final_Q_table.loc [:,'GRID_ON'])})
#q_table_diff_heat = pd.DataFrame({'q_table_diff': (heat_q_table.loc [:,'GRID_OFF'] - heat_q_table.loc [:,'GRID_ON'])})
#        
## Create the 3 heat maps
#Table_ON = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#Table_OFF = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#Table_diff = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#
#Table_ON_heat = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#Table_OFF_heat = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#Table_diff_heat = pd.DataFrame(0, index=np.arange((n_SOC-1)*dp,-dp,-dp), columns=np.arange(Pnet_min,Pnet_max+dp,dp))
#
#for i in range ((n_SOC-1)*int(dp),int(-dp),int(-dp)):
#    for j in range (int(Pnet_min),int(Pnet_max)+int(dp),int(dp)):
#        
#        i_time = int(round(Temp_affich_heatmap / dt))
#        i_soc = int(round((i - SoC_min) / dp))
#        i_Pnet = int(round((j - Pnet_min) / dp))
#        index_heat = n_Pnet * (i_time * n_SOC + i_soc) + i_Pnet
#        
#        Table_ON.loc[i,j] = Final_Q_table.loc[index_heat,'GRID_ON']
#        Table_OFF.loc[i,j] = Final_Q_table.loc[index_heat,'GRID_OFF']
#        Table_diff.loc[i,j]= q_table_diff.loc[index_heat,'q_table_diff']
#        
#for i in range ((n_SOC-1)*int(dp),int(-dp),int(-dp)):
#    for j in range (int(Pnet_min),int(Pnet_max)+int(dp),int(dp)):
#        
#        i_time = int(round(Temp_affich_heatmap / dt))
#        i_soc = int(round((i - SoC_min) / dp))
#        i_Pnet = int(round((j - Pnet_min) / dp))
#        index_heat = n_Pnet * (i_time * n_SOC + i_soc) + i_Pnet
#        
#        Table_ON_heat.loc[i,j] = heat_q_table.loc[index_heat,'GRID_ON']
#        Table_OFF_heat.loc[i,j] = heat_q_table.loc[index_heat,'GRID_OFF']
#        Table_diff_heat.loc[i,j]= q_table_diff_heat.loc[index_heat,'q_table_diff'] 
        
# %% boxplot preparation:
performence_slice = split_list(performence, wanted_parts=int(n_episode/delta_episode_performence))
performence_slice = np.transpose(performence_slice)
performence_slice = pd.DataFrame(performence_slice , columns=(np.arange(n_episode/delta_episode_performence)+1) )

#%% xticks preparation:

freq_heur_affich = 6
step_xtick= 2*freq_heur_affich # must put a value twice as much as freq_affich
freq_affich = str(freq_heur_affich)+'H'
rng = pd.Series(pd.date_range(months_of_train[0], freq=freq_affich, periods = n_jour_visio*4))
rng=pd.DataFrame({'rng': rng})
rng=rng['rng'].astype(str)
out = [x[11:-3] for x in rng]

# %% Display

plt.figure(1)
plt.subplot(211)
plt.plot(Pnet1_last_day)
plt.plot(P_reel_last_day)
plt.plot(Pgrid_last_day)
plt.plot(Pprod_shed_step_last_day)
plt.plot(Pcons_unsatisfied_step_last_day)
plt.legend(('planned net demand power', 'realized net demand power with ' + str(teau_incertitude) + ' percent uncertainty', 'Pgrid', 'Pprod_shed', 'Pcons_unsatisfied'), loc='best')
plt.xlabel('hour')
plt.ylabel('Power (W)')
plt.xticks(ticks = np.arange(0,len(Statofcharge_last_day),12), labels = out)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Statofcharge_last_day)
plt.xlabel('hour')
plt.ylabel('Energy (Wh)')
plt.xticks(ticks = np.arange(0,len(Statofcharge_last_day),step_xtick), labels = out)
plt.title('Energy Management Report By applying RL for last "' + str(n_jour_visio) + '" days of Training process')
plt.grid(True)
plt.show()


plt.figure(2)
performence_slice.boxplot(grid=True)
plt.xlabel('Number of times the Q_table was applied for " '+ str(nombr_jour_boxplot)+' " days')
plt.ylabel('Average cost per group of " '+ str(nombr_jour_boxplot)+' "days / [Euros]')
plt.title('Performance over " '+ str(nombr_jour_boxplot)+ ' " successive days')
plt.show()

#plt.figure(3)
#plt.plot(reward_episode,'yo-')
#plt.xlabel('episodes')
#plt.ylabel('cumulative rewards per episode')
#plt.title('Penalty Per episode')
#plt.show()
#plt.grid(True)

#plt.figure(4) 
#sns.heatmap(Table_ON, annot=False , cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Grid_ON at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()

#plt.figure(5) 
#sns.heatmap(Table_OFF, annot=False, cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Grid_OFF at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()

#plt.figure(6)
#sns.heatmap(Table_ON_heat, vmin=0, vmax=1 , annot= False , cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Grid_ON at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()

#plt.figure(7) 
#sns.heatmap(Table_OFF_heat, vmin=0, vmax=1 , annot= False  , cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Grid_OFF at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()

#plt.figure(8) 
#sns.heatmap(Table_diff , annot=False, cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Diff_ON_OFF at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()

#plt.figure(9)  
#sns.heatmap(Table_diff_heat , vmin=0, vmax=1 , annot= False , cmap="YlGnBu", linewidths=.3)
#plt.xlabel('Pnet')
#plt.ylabel('SOC')
#plt.title('Q_table_Diff_ON_OFF at "'+ str(Temp_affich_heatmap) + '" hour')
#plt.show()
# %% Save the Q_table
data_2write = pd.concat([Final_Q_table['GRID_OFF'],Final_Q_table['GRID_ON'] ],axis=1, join='inner', keys=['GRID_OFF','GRID_ON'] , sort=False)
info_2write = pd.DataFrame({'dp':[dp], 'dt':[dt], 'Pnet_min_q_table': [Pnet_min], 'Pnet_max_q_table' : [Pnet_max], 'SoC_min': [SoC_min], 'SoC_max':[SoC_max]}) 
file_write_1 = 'Q_table_Predict_' + str(n_episode) +' episodes_'+ str(teau_incertitude) + ' percent incertity'
file_write = file_write_1 + '.txt'
info_write = 'information_' + file_write_1 + '.txt'
sep_write = '\t'
path_write = r"C:\Users\mdini\Documents\GitHub\microgrid_ai\data\drahix"
data_2write.to_csv(os.path.join(path_write,file_write),sep=sep_write,index=True)
info_2write.to_csv(os.path.join(path_write,info_write),sep=sep_write,index=True)
