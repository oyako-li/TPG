import numpy as np
import matplotlib.pyplot as plt

with open("log/Acrobot-v1/2022-06-20_20-01-14.log", "r") as file:
    last_line = file.readlines()[-1]
    result = ','.join(last_line.split(',')[7:])
    result = result[3:-3].split('), (')
    l = []
    for item in result:
        _min, _max, _ave = item.split(', ')
        l.append((float(_min), float(_max), float(_ave)))
    
    __min = []
    __mi = 0.
    __max = []
    __ma = 0.
    __ave = []
    __av = 0.
    for i, item in enumerate(l):
        _min, _max, _ave = item
        __mi+=_min
        __ma+=_max
        __av+=_ave
        if i%5==0:
            __min.append(__mi)
            __mi = 0.
            __max.append(__ma)
            __ma = 0.
            __ave.append(__av)
            __av = 0.
    mi = np.array(__min)
    ma = np.array(__max)
    av = np.array(__ave)

    ge = np.linspace(5, 1000, mi.size)
    print(l)