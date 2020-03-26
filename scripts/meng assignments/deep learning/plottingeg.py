
"""
@author: rlk268@cornell.edu
"""

#assume you have data loaded already, have access to meas and platooninfo

from havsim.plotting import plotvhd_v2
import matplotlib.pyplot as plt

#plotting speed
plt.figure()
plt.plot(meas[1013][:,3]) 

plotvhd_v2(meas, None, platooninfo, [1013])

plotvhd_v2(meas,None, platooninfo, [1013, 17])
