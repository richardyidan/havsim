
"""
@author: rlk268@cornell.edu
"""

import numpy as np
import time
# import multiprocessing
import pathos.multiprocessing as pm

def blah(a,b):
    return a+b

class toyVehicle:
    def __init__(self):
        self.rand = np.random.rand()
        self.res = None

    def test(self, a, b):
        temp = a**(b*self.rand+a**b+(a+b)/b)
        self.res = temp**(b*self.rand+b**a)


def ptest():
    vehicles = [toyVehicle() for i in range(10000)]
    a = 2
    b = 1.5
    start = time.time()
    out = [vehicle.test(a,b) for vehicle in vehicles]
    res = sum([vehicle.res for vehicle in vehicles])
    end = time.time()
    print(str(end-start)+' to find result '+str(res))
    return res

out = ptest()

# def ptest2():
#     vehicles = [toyVehicle() for i in range(3)]
#     a = 2
#     b = 1.5
#     start = time.time()
#     queue = multiprocessing.SimpleQueue()
#     # if __name__ == '__main__':
#     with multiprocessing.Pool() as pool:
#     # pool = multiprocessing.Pool()
#         # res = pool.apply_async(blah, (a,b))
#         # # print(res.get(timeout = 1))
#         # out = res.get(timeout = 1)
#         for vehicle in vehicles:
#             res = pool.apply_async(vehicle.test, (a,b))
#             queue.put(vehicle)
#             # out = res.get(timeout = 1)
#     vehicles = [queue.get() for i in queue]
#     res = sum([vehicle.res for vehicle in vehicles])
#     end = time.time()
#     print(str(end-start)+' to find result '+str(res))
#     return out

def wrapper(args):
    veh, a, b = args
    veh.test(a,b)
    return veh


def ptest2():
    a,b = 2, 1.5
    vehicles = [toyVehicle() for i in range(10000)]
    start = time.time()
    starargs = ((veh,a,b) for veh in vehicles)

    with pm.Pool() as pool:
        vehicles = pool.map(wrapper, starargs)

    res = sum(veh.res for veh in vehicles)
    print(str(time.time()-start)+' to find result '+str(res))
    return res

def ptest2():
    a,b = 2, 1.5
    vehicles = [toyVehicle() for i in range(10000)]
    start = time.time()
    starargs = ((veh,a,b) for veh in vehicles)

    with pm.Pool() as pool:
        vehicles = pool.map(wrapper, starargs)

    res = sum(veh.res for veh in vehicles)
    print(str(time.time()-start)+' to find result '+str(res))
    return res

if __name__ == '__main__':
    out = ptest2()
# a,b = 2, 1.5
# if __name__ == '__main__':
#     with multiprocessing.Pool() as pool:
#         res = pool.apply_async(blah, (a,b))
#         print(res.get(timeout = 1))


def mytimer(funtotime, args = None):
    start = time.time()
    if args is not None:
        funtotime(*args)
    else:
        funtotime()
    end= time.time()
    print(str(end-start))