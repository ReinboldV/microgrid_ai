# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:10:12 2020

@author: mdini
"""

import math  
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
###### Récuperer les données historiques de Pnet = Pload - Ppv 

with open("Profil_4.csv", newline='') as csvfile :
    reader = csv.reader(csvfile)
    liste = []
    for row in reader:
        liste += row
n_points = len(liste)
Pnet1 = np.array([0.]*n_points)
for r in range(n_points) :
    Pnet1[r] = float(liste[r])
    
Pnet=Pnet1 

###### Produir des données Pnet par hazard:

#Pnet=np.zeros(24)
#for i in range(24):    
#    Pnet[i]=math.floor(np.random.randint(-2,3))*100

################################################################

E_bat_max = 1000    # the maximum value
E_bat_min = 0       # the minimum value
dt = 1
E_bat = [0]*(len(Pnet))
P_grid = [0]*(len(Pnet))
E_pv_shed = [0]*(len(Pnet))
E_bat[0] = 500
for i in range(len(E_bat)-1):
    E_bat[i+1] = E_bat[i]-Pnet[i]*dt
    if E_bat[i+1] < E_bat_min:
        E_bat[i+1] = E_bat[i]
        P_grid[i] = Pnet[i]
    else:
        if E_bat[i+1] > E_bat_max:
            E_pv_shed[i] =  E_bat_max - E_bat[i+1] 
            E_bat[i+1] = E_bat_max
            
E_bat[-1] = E_bat[-2]-Pnet[-1]*dt
if E_bat[-1] <E_bat_min:
            E_bat[-1] = E_bat[-2]
            P_grid[-1] = Pnet[-1]
else:
    if E_bat[-1] > E_bat_max:
        E_pv_shed[-1] = E_bat_max -E_bat[-1]   
        E_bat[-1] = E_bat_max
        
########################################################################################        
#for i in range (t_fin-1):
#    if Pnet[i]<=0:                    # PV power is greater than users' demand (enough production)                      
#        Pb[i]=Pnet[i]
#        SOC[i+1]=SOC[i]-Pb[i]
#        Pdg[i]=0                      # turn off the diesel generator
#        if SOC[i+1]>=SOCmax:               
#            SOC[i+1]=SOCmax
#            Pshed_PV[i+1]=SOC[i+1]-SOCmax
#            
#    if Pnet[i]> 0:                    #PV power is smaller than users' demand (not enough production)
#        Pdg[i]=Pnet[i]
#        SOC[i+1]=SOC[i]       
#
#    Pdg[i+1]=Pdg[i]                   # keep the DG state for the next step
############################# fee ######################################################
Cout_grid = 10
Cout_pv = 10 
C_grid = np.cumsum(P_grid)
C_pv_shed=np.cumsum(E_pv_shed)

Ctotal =Cout_grid  * C_grid[-1] - Cout_pv * C_pv_shed[-1]
print('Cout_Total = ',Ctotal)
########################################################################################

plt.figure(1)
plt.plot(Pnet,'ro-') 
plt.xlabel('Houre')
plt.ylabel('Pnet')
plt.title('Pnet')
plt.show()
plt.grid(True)

plt.figure(2)
plt.plot(E_bat,'bo-')
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('Ebat')
plt.grid(True)

plt.figure(3)
plt.plot(P_grid,'go-')
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('P_grid')
plt.grid(True)



plt.figure(4)
plt.subplot(211)
plt.plot(E_bat )
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('Bilan_RB')
plt.legend(('Statofcharge'), loc='best', shadow=True)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Pnet )
plt.plot(P_grid)
plt.plot(E_pv_shed)
plt.legend(('Pnet', 'Pgrid ','Pprod_shed'), loc='best', shadow=True)
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.grid(True)
plt.show()


labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet))  # the label location
fig, axs = plt.subplots(2, 1)
axs[0].bar(x             ,E_bat     , 2*width  , label='E_bat')
axs[1].bar(x  - width    ,Pnet      , width    , label='Pnet')
axs[1].bar(x             ,P_grid    , width    , label='P_grid') 
axs[1].bar(x + width     ,E_pv_shed , width    , label='Pprod_shed')
fig.suptitle('Bilan_RB')
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