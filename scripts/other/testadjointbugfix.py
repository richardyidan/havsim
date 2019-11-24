
"""
@author: rlk268@cornell.edu


"""

#pguess = [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ] #this seems like a very good second guess
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ] #original guess
#pguess = [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]

mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]

args = (True, 6)