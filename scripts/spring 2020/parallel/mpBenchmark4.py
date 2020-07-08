
"""
@author: rlk268@cornell.edu
"""

from multiprocessing import Pool
import time

def f(x):
    return x*x*x*x

if __name__ == '__main__':
    p = Pool(5)
    l = [x for x in range(2000000)]
    start = time.clock()
    p.map(f, l)
    end = time.clock()
    print('pool processing time {}'.format(end - start))
    start = time.clock()
    map(f, l)
    end = time.clock()
    print('sequential processing time {}'.format(end - start))