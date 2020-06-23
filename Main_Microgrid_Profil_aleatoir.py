# -*- coding: utf-8 -*-

"""
Created on Fri May 15 10:30:13 2020

@author: mdini
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.optimize import curve_fit
from agent_microgrid import Agent
from microgrid_simulator import MicrogridSimulator
plt.close('all')



##### Nombre d'épisode pour training
n_points=24    
n_episode= 100            # Initialisation de nombre d'épisodes 
n_actions= 2                               # Initialisation de la taille d'actions possible(Grid_On ou Grid_OFF)
  
###### Produir des données Pnet par hazard:   
Pnet1=np.zeros(n_points*n_episode)
for i in range(n_points*n_episode):    
    Pnet1[i]=math.floor(np.random.randint(-2,3))*100
                           

##### Initialisation de l'environement et l'espace d'état
dP, n_SOC, n_pmax = 100, 11, 2

SOC = np.arange(n_SOC)*dP
n_Pnet = 2*n_pmax+1                        # number of power levels (power ranges between -n_power*dP and +n_power*dP)
Pnet = np.arange(-n_pmax,n_pmax+1)*dP


n_state = n_SOC*n_Pnet                     # Initialisation de la taille de l'espace d'état 
state_SOC = np.zeros(n_state)
state_Pnet = np.zeros(n_state)   

# Les coefficients de Rewards:
Cbatful, Cbat_empty,  Cinject_grid , Cgrid_use , Cbat_use =  50, 1, 50, 5, 0

microgrid = MicrogridSimulator(dP, n_SOC, n_pmax, Cbatful, Cinject_grid, Cbat_empty , Cgrid_use  , Cbat_use )


#### Initialisation et Choisir agent parmi(Agent(Q-learning), Accountant, Drunkard)

learning_rate=0.9  
discount=0.95
exploration_rate=1 
iterations=n_episode*n_points

agent=Agent(n_state, n_actions, learning_rate, discount, exploration_rate , iterations)

################################################################################################################  
##### main loop    
reward_episode = np.zeros(n_episode)

Pgrid_episode = np.zeros(n_episode)


Pgrid_step = np.zeros(n_episode*n_points)
Pprod_shed_step=np.zeros(n_episode*n_points)
Pcons_unsatisfied_step=np.zeros(n_episode*n_points)


last_total_diff=np.zeros(n_episode)
last_total_diff[0]=0

performance=np.zeros(n_episode)

Statofcharge=np.zeros(iterations)


Statofcharge_last_day= np.zeros(n_points)
Pnet1_last_day = np.zeros(n_points)
Pgrid_last_day = np.zeros(n_points)
Pprod_shed_step_last_day= np.zeros(n_points)
Pcons_unsatisfied_step_last_day= np.zeros(n_points)
    

Statofcharge_first_day= np.zeros(n_points)
Pnet1_first_day = np.zeros(n_points)
Pgrid_first_day = np.zeros(n_points)
Pprod_shed_step_first_day= np.zeros(n_points)
Pcons_unsatisfied_step_first_day= np.zeros(n_points)


Statofcharge_middle_day= np.zeros(n_points)
Pnet1_middle_day = np.zeros(n_points)
Pgrid_middle_day = np.zeros(n_points)
Pprod_shed_step_middle_day= np.zeros(n_points)
Pcons_unsatisfied_step_middle_day= np.zeros(n_points)





for episode in range (n_episode):
    total_reward=0
    total_Pgrid=0                                 
    for step in range(n_points):
        old_state = microgrid.state
        action = agent.get_next_action(old_state)                     # Query agent for the next action
        new_state, reward ,Pgrid,Pprod_shed,Pcons_unsatisfied = microgrid.take_action(action,Pnet1[(episode*n_points)+step]) # Take action, get new state and reward
        agent.update(old_state, new_state, action, reward)            # Let the agent update internals  
        Pgrid_step[(episode*n_points)+step] = Pgrid
        Pprod_shed_step[(episode*n_points)+step] = Pprod_shed
        Pcons_unsatisfied_step[(episode*n_points)+step] = Pcons_unsatisfied
        
        total_reward += reward  
        total_Pgrid += Pgrid
        Statofcharge[(episode*n_points)+step]=microgrid.state_SOC[new_state] 
        

        
    Pgrid_episode[episode] = total_Pgrid
    
    reward_episode[episode] = total_reward
                                     
    performance[episode] = (reward_episode[episode]-last_total_diff[episode-1])/n_points/100
    
    last_total_diff[episode] = reward_episode[episode]
     
    
for i in range(n_points):
    Statofcharge_last_day[i] = Statofcharge[((n_episode-1)*n_points)+i]
    Pnet1_last_day[i] = Pnet1[((n_episode-1)*n_points)+i]
    Pgrid_last_day[i-1]=Pgrid_step[((n_episode-1)*n_points)+i]
    Pprod_shed_step_last_day[i-1]=Pprod_shed_step[((n_episode-1)*n_points)+i]
    Pcons_unsatisfied_step_last_day[i-1]=Pcons_unsatisfied_step[((n_episode-1)*n_points)+i]
    
    Statofcharge_first_day[i] = Statofcharge[i]
    Pnet1_first_day[i]=Pnet1[i]
    Pgrid_first_day[i]=Pgrid_step[i]
    Pprod_shed_step_first_day[i]=Pprod_shed_step[i]
    Pcons_unsatisfied_step_first_day[i]=Pcons_unsatisfied_step[i]
    
    Statofcharge_middle_day[i] = Statofcharge[int((n_episode-(0.5*n_episode)))*n_points+i]
    Pnet1_middle_day[i]=Pnet1[int((n_episode-(0.5*n_episode)))*n_points+i]
    Pgrid_middle_day[i]=Pgrid_step[int((n_episode-(0.5*n_episode)))*n_points+i]
    Pprod_shed_step_middle_day[i]=Pprod_shed_step[int((n_episode-(0.5*n_episode)))*n_points+i]
    Pcons_unsatisfied_step_middle_day[i]=Pcons_unsatisfied_step[int((n_episode-(0.5*n_episode)))*n_points+i] 
    
print("Final Q-table", agent.q_table )  
#####################################################################################################
Q_table=agent.q_table 
#
#for i in range (n_state):
#    if(Q_table[i][0] < Q_table[i][1]): 
#        Q_table[i][0]=0
#        Q_table[i][1]=1
#    else:
#        Q_table[i][0]=1
#        Q_table[i][1]=0   
#        
#print("Final Q-table", agent.q_table)

Javab=np.zeros([n_SOC+1, n_Pnet+1] )
for i in range (1,n_SOC+1):
    Javab[i][0]=100*i-100
for i in range (1,n_Pnet+1):   
    Javab[0][i]=100*i-300
    
for i in range (1,n_SOC+1):
    for j in range (1,n_Pnet+1): 
        Javab[i][j]=Q_table[i][0]
print(Javab)
data=Javab
plt.figure(0) 
sns.heatmap(data, annot=True,  linewidths=.5)
##################################################################################################
### fee    
Cout_grid = 10
C_pv_shed = 5
C_conso_unsatisfied= 10

Pgrid_out=np.cumsum(Pgrid_last_day)
Pprod_shed_out=np.cumsum(Pprod_shed_step_last_day)
Pcons_unsatisfied_out=np.cumsum(Pcons_unsatisfied_step_last_day)

Ctotal = Cout_grid * Pgrid_out[-1] - C_pv_shed *Pprod_shed_out[1] - C_conso_unsatisfied * Pcons_unsatisfied_out[-1]
print('Cout_Total_Last_day = ',Ctotal)
#####################################################################################################
plt.figure(1) 
plt.plot(reward_episode,'yo-')  
plt.xlabel('episods')
plt.ylabel('cumulative rewards per episode')
plt.title('Reward_Par_episode')
plt.show()
plt.grid(True)

##plt.figure(2)
##plt.plot(Pnet1,'ro-') 
##plt.xlabel('Houre')
##plt.ylabel('Pnet')
##plt.title('Pnet_chaque_instant')
##plt.show()
##plt.grid(True)

##plt.figure(3)
##plt.plot(Statofcharge,'bo-')
##plt.xlabel('Steps(hours)= episode*length database')
##plt.ylabel('SoC')
##plt.title('SoC_Pendant_training') 
##plt.grid(True)

##plt.figure(4) 
##plt.plot(Pgrid_episode,'go-')
##plt.plot(Pgrid_episode[-1]*10,'ro-')  
##plt.xlabel('steps')
##plt.ylabel('Pgrid')
##plt.title('Pgrid_Par_episode')
##plt.show()
##plt.grid(True)

##plt.figure(5)
##plt.plot(performance,'bo-') 
##plt.xlabel('episods')
##plt.ylabel('Actuel reward  - Last reward')
##plt.title('Difference reward between two episodes')
##plt.show()
##plt.grid(True)

#plt.figure(6)
#plt.plot(Statofcharge_first_day )
#plt.plot(Statofcharge_middle_day )
#plt.plot(Statofcharge_last_day )
#plt.legend(('Statofcharge_first_day','Statofcharge_middle_day','Statofcharge_last_day'), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('StateofCharge')
#plt.grid(True)



#plt.figure(7)
#plt.plot(Pnet1_first_day )
#plt.plot(Pnet1_middle_day )
#plt.plot(Pnet1_last_day )
#plt.legend(('Pnet_first_day', 'Pnet_middle_day', 'Pnet_last_day'), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('Pnet')
#plt.grid(True)


#plt.figure(8)
#plt.plot(Pgrid_first_day )
#plt.plot(Pgrid_middle_day )
#plt.plot(Pgrid_last_day )
#plt.legend(('Pgrid_first_day ', 'Pgrid_middle_day ','Pgrid_last_day '), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('Pgrid')
#plt.grid(True)




#plt.figure(9)
#plt.plot(Pprod_shed_step_first_day )
#plt.plot(Pprod_shed_step_middle_day )
#plt.plot(Pprod_shed_step_last_day )
#plt.legend(('Pprod_shed_first_day ', 'Pprod_shed_middle_day ','Pprod_shed_last_day '), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('Pprod_shed')
#plt.grid(True)



#plt.figure(10)
#plt.plot(Pcons_unsatisfied_step_first_day )
#plt.plot(Pcons_unsatisfied_step_middle_day )
#plt.plot(Pcons_unsatisfied_step_last_day )
#plt.legend(('Pcons_unsatisfied_first_day ', 'Pcons_unsatisfied_middle_day ','Pcons_unsatisfied_last_day '), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('Pcons_unsatisfied')



#plt.figure(11)
#plt.subplot(211)
#plt.plot(Statofcharge_last_day )
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.title('Bilan_Last_Day_RL')
#plt.legend(('Statofcharge_last_day'), loc='best', shadow=True)
#plt.grid(True)
#plt.show()
#plt.subplot(212)
#plt.plot(Pnet1_last_day)
#plt.plot(Pgrid_last_day )
#plt.plot(Pprod_shed_step_last_day )
#plt.plot(Pcons_unsatisfied_step_last_day)
#plt.legend(('Pnet_last_day', 'Pgrid_last_day ','Pprod_shed_step_last_day','Pcons_unsatisfied_step_last_day'), loc='best', shadow=True)
#plt.xlabel('Houre')
#plt.ylabel('Puissance')
#plt.grid(True)
#plt.show()

###############################################################################################################################################
labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(1, 1)
axs.bar(x  - width    ,Statofcharge_first_day         , width  , label='Statofcharge_first_day ')
axs.bar(x             ,Statofcharge_middle_day        , width  , label='Statofcharge_middle_day') 
axs.bar(x + width     ,Statofcharge_last_day          , width  , label='Statofcharge_last_day')
fig.suptitle('Statofcharge')
axs.set_ylabel('Puissance')
axs.set_xlabel('Houre')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend()

labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(1, 1)
axs.bar(x  - width    ,Pnet1_first_day        , width  , label='Pnet1_first_day')
axs.bar(x             ,Pnet1_middle_day        , width  , label='Pnet1_middle_day ') 
axs.bar(x + width     ,Pnet1_last_day           , width  , label='Pnet1_last_day ')
fig.suptitle('Pnet')
axs.set_ylabel('Puissance')
axs.set_xlabel('Houre')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend()


labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(1, 1)
axs.bar(x  - width    ,Pgrid_first_day         , width  , label='Pgrid_first_day')
axs.bar(x             ,Pgrid_middle_day        , width  , label='Pgrid_middle_day') 
axs.bar(x + width     ,Pgrid_last_day          , width  , label='Pgrid_last_day')
fig.suptitle('Pgrid')
axs.set_ylabel('Puissance')
axs.set_xlabel('Houre')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend()





labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(1, 1)
axs.bar(x  - width    ,Pprod_shed_step_first_day        , width  , label='Pprod_shed_step_first_day')
axs.bar(x             ,Pprod_shed_step_middle_day        , width  , label='Pprod_shed_step_middle_day') 
axs.bar(x + width     ,Pprod_shed_step_last_day           , width  , label='Pprod_shed_step_last_day ')
fig.suptitle('Pprod_shed')
axs.set_ylabel('Puissance')
axs.set_xlabel('Houre')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend() 

labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location
fig, axs = plt.subplots(1, 1)
axs.bar(x  - width    ,Pcons_unsatisfied_step_first_day        , width  , label='Pcons_unsatisfied_step_first_day')
axs.bar(x             ,Pcons_unsatisfied_step_middle_day       , width  , label='Pcons_unsatisfied_step_middle_day') 
axs.bar(x + width     ,Pcons_unsatisfied_step_last_day         , width  , label='Pcons_unsatisfied_step_last_day')
fig.suptitle('Pcons_unsatisfied')
axs.set_ylabel('Puissance')
axs.set_xlabel('Houre')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend()



labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet1_last_day))  # the label location


fig, axs = plt.subplots(2, 1)
axs[0].bar(x             ,Statofcharge_last_day          ,2*width, label='Statofcharge_last_day' )

axs[1].bar(x  - width    ,Pnet1_last_day                 , width  , label='net_last_day')
axs[1].bar(x             ,Pgrid_last_day                 , width  , label='Pgrid_last_day') 
axs[1].bar(x + width     ,Pprod_shed_step_last_day       , width  , label='Pprod_shed_last_day')
axs[1].bar(x + (2*width) ,Pcons_unsatisfied_step_last_day, width  , label='Pcons_unsatisfied_last_day')
fig.suptitle('Bilan_Last_Day_RL')
axs[0].set_ylabel('Puissance')
axs[0].set_xlabel('Houre')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].legend()
axs[0].grid(True)
axs[1].set_ylabel('Puissance')
axs[1].set_xlabel('Houre')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].legend()
axs[1].grid(True)