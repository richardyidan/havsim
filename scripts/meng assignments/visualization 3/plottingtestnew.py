
"""
@author: rlk268@cornell.edu
"""
#imports 
import pickle 
import matplotlib.pyplot as plt
from havsim.plotting import platoonplot, calculateflows, plotflows

#load data 
with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/meng/plottingtesting.pkl','rb') as f:
     meas, platooninfo, platoonlist, sim = pickle.load(f)

#with open('D:/assignment/Meng/plottingtesting.pkl','rb') as f:
#    meas, platooninfo, platoonlist, sim = pickle.load(f)
     
    
#test on actual data
plt.close('all')
plt.figure()
plotflows(meas,[[400,600], [800, 1000]], [1000, 3000], 300, 'line')
plt.figure()
plotflows(meas,[[400,600], [800, 1000]], [1000, 3000], 300, 'FD')

"""
TO DO 
The above functions throw errors; fix
"""   
    
    
#%%
#test on toy example 
import numpy as np
testmeas = {}
for i in range(1):
    testmeas[i] = np.zeros((1001,3))
    testmeas[i][:,1] = np.linspace(0,1000,1001)
    testmeas[i][:,2] = np.linspace(0,1000,1001)
 #3 vehicles, all of them have s = 0-1000. the times are 0-1000 for vehicle 0
plt.figure()
plotflows(testmeas,[[200,400],[800,1000]],[0,1000],300,'line')
#first line should be made up of points (100/60000, 100/60000, 0, 0) on times (0,300,600,900)
#second line should be made up of points (0, 0, 100/60000,100/20000)


testmeas2 = {}
for i in range(3):
    testmeas2[i] = np.zeros((1001,3))
    testmeas2[i][:,1] = np.linspace(0+100*i,1000+100*i,1001) #equivalent to list(range(1001+100*i))[100*i:]
    testmeas2[i][:,2] = np.linspace(0,1000,1001)

"""
TO DO
 figure out what the right values should be for the below example and verify that the function is giving the correct values.
 record what the right values are and show your work. 
"""
plt.figure()
plotflows(testmeas2, [[200,400],[800,1000]],[0,1000],300,'line')

#%%
#another toy example I want you to test on to verify on
meas[898][100:200,7] = 3
meas[905][150:250,7] = 3
platoonplot(meas,None,platooninfo,platoon=[[],898, 905, 909, 916, 920], lane=2,opacity =.1, colorCode= True, speed_limit = [10,35]) 
plt.plot([2600, 2600, 2800, 2800, 2600], [400, 800, 800, 400, 400], 'k-')
plt.plot([2800, 2800, 3000, 3000, 2800], [400, 800, 800, 400, 400], 'k-')

q, k = calculateflows(testmeas2, [[400, 800]], [2600, 3000], 200)
"""
TO DO 
add feature where you can select only a particular lane to take the flow on. IN this case if a vehicle is not in the specified lane 
it will not count towards the flow and density counts. 
Again record what the right values are and show your work. 
"""
#%%
vehlist = []
for i in platoonlist: 
    vehlist.extend(i[1:])
platoonplot(meas,sim,platooninfo, vehlist, lane=2,opacity =.1, colorCode= False, speed_limit = [10,35]) 
