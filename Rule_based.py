"""
Created on Tue Jun  9 14:06:30 2020

@author: mdini
"""
import math  
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
###### Récuperer les données historiques de Pnet = Pload - Ppv 

#with open("Profil_3.csv", newline='') as csvfile :
#    reader = csv.reader(csvfile)
#    liste = []
#    for row in reader:
#        liste += row
#n_points = len(liste)
#Pnet1 = np.array([0.]*n_points)
#for r in range(n_points) :
#    Pnet1[r] = float(liste[r])
#    
#Pnet=Pnet1 

###### Produir des données Pnet par hazard:
Pnet=np.zeros(24)
for i in range(24):    
    Pnet[i]=math.floor(np.random.randint(-2,3))*100
    
##########################################################     
t_fin= len(Pnet)
Pb=np.zeros([t_fin])
Pshed_PV=np.zeros([t_fin])
Pshed_PV[0]=0
Pl_out=np.zeros([t_fin])
Pdg=np.zeros([t_fin])
Pdg[0]=0 

SOC=np.zeros([t_fin])
SOC[0]=500      # initial value
SOCmax= 1000    # the maximum value
SOCmin= 0       # the minimum value


######## Le cas où on préfer d'abord utiliser la batterie jusqu'à ses limites, quand on a la surconsomation ###########

for i in range (t_fin-1):    
    if Pnet[i] <= 0:                      # PV power is greater than users' demand (enough production)                      
        Pb[i]=Pnet[i]
        SOC[i+1]=SOC[i]-Pb[i]
        Pdg[i]=0                          # turn off the diesel generator
        if SOC[i+1] >= SOCmax:               
            SOC[i+1] = SOCmax
            Pshed_PV[i+1] = SOC[i+1]-SOCmax
              
            
    if Pnet[i]> 0:                        #PV power is smaller than users' demand (not enough production)
        Pb[i] = Pnet[i]
        SOC[i+1] = SOC[i] - Pb[i]        
        if SOC[i+1] <= SOCmin:
            Pdg[i] = SOCmin - SOC[i+1]    #turn on the diesel generator
            SOC[i+1] = SOCmin
            
    Pdg[i+1]=Pdg[i]                       # keep the DG state for the next step
    
    
######### Le cas où on ne touche pas la batterie quand on a la surconsomation ############

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

##################################################################################################  
### fee
Cout_dg = 10
Cdg = np.cumsum(Pdg)
Ctotal = Cout_dg * Cdg[-1]
print('Cout_Total = ',Ctotal)
#################################################################################################

plt.figure(1)
plt.plot(Pnet,'ro-') 
plt.xlabel('Houre')
plt.ylabel('Pnet')
plt.title('Pnet')
plt.show()
plt.grid(True)

plt.figure(2)
plt.plot(SOC,'bo-')
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('State of Charge')
plt.grid(True)

plt.figure(3)
plt.plot(Pdg,'go-')
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('Puissance de DG')
plt.grid(True)


plt.figure(4)
plt.subplot(211)
plt.plot(SOC )
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.title('Bilan_RB')
plt.legend(('Statofcharge'), loc='best', shadow=True)
plt.grid(True)
plt.show()
plt.subplot(212)
plt.plot(Pnet )
plt.plot(Pdg)
plt.legend(('Pnet', 'Pgrid ','Pprod_shed'), loc='best', shadow=True)
plt.xlabel('Houre')
plt.ylabel('Puissance')
plt.grid(True)
plt.show()


labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
width = 0.2
x = np.arange(len(Pnet))  # the label location
fig, axs = plt.subplots(2, 1)
axs[0].bar(x             ,SOC     , width  , label='SOC')
axs[1].bar(x  - width    ,Pnet    , width  , label='Pnet')
axs[1].bar(x             ,Pdg     , width  , label='Pdg') 
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