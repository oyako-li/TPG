import logging
import time
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def setup_logger(_name, _logfile='LOGFILENAME', test=False, load=True):
    
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.DEBUG)

    _filename = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(f'{_logfile} {_filename}')
    time.sleep(0.5)
    # create file handler which logs even DEBUG messages
    if not test:
        while True:
            try:
                _fh = logging.FileHandler(f'log/{_logfile}/{_filename}.log')
                break
            except FileNotFoundError:
                os.makedirs(f'log/{_logfile}')

        _fh.setLevel(logging.INFO)
        _fh_formatter = logging.Formatter('%(asctime)s, %(filename)s:%(className)s, %(message)s')
        _fh.setFormatter(_fh_formatter)
        _logger.addHandler(_fh)


    # create console handler with a INFO log level
    if load:
        _ch = logging.StreamHandler()
        _ch.setLevel(logging.DEBUG)
        _ch_formatter = logging.Formatter('[{}][{}]%(name)s,%(className)s:%(message)s'.format(_logfile, _filename))
        _ch.setFormatter(_ch_formatter)

        # add the handlers to the logger
        _logger.addHandler(_ch)
    return _logger, _filename

def log_load(_filename, _renge, _step=5):
    l =[]

    with open(f"{_filename}.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            results = line.replace('\n','').split(', ')[3:]
            l.append([float(re.split(':')[1]) for re in results])

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
        # if i==0: continue
        if i%_step==_step-1:
            __min.append(__mi/float(_step))
            __mi=0.
            __max.append(__ma/float(_step))
            __ma=0.
            __ave.append(__av/float(_step))
            __av=0.
        if i == _renge: break
    mi = np.array(__min)
    ma = np.array(__max)
    av = np.array(__ave)
    
    return mi, ma, av

def log_show(filename, renge=100, step=5):
    mi, ma, av = log_load(filename, renge, step)
    ge = np.linspace(step, renge, mi.size)
    # Figure instance
    fig = plt.Figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(ge, mi, label='min')
    ax1.plot(ge, ma, label='max')
    ax1.plot(ge, av, label='ave')
    ax1.set_title(f'{filename}')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Generation')
    ax1.legend()

    # # ax2
    # ax2 = fig.add_subplot(222)
    # ax2.plot(ge, ma)
    # ax2.set_title('Scatter plot')

    # # ax3
    # ax3 = fig.add_subplot(223)
    # ax3.plot(ge, av)
    # ax3.set_ylabel('Damped oscillation')
    # ax3.set_xlabel('time (s)')
    # When windows is closed.

    def _destroyWindow():
        fig.savefig(f'{filename}.png')
        root.quit()
        root.destroy()



    # Tkinter Class

    root = tk.Tk()
    root.withdraw()
    root.protocol('WM_DELETE_WINDOW', _destroyWindow)  # When you close the tkinter window.

    # Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)  # Generate canvas instance, Embedding fig in root
    canvas.draw()
    canvas.get_tk_widget().pack()
    #canvas._tkcanvas.pack()

    # root
    root.update()
    root.deiconify()
    root.mainloop()

def log_load2(_filename, _renge, _step=5):
    l =[]

    with open(f"{_filename}.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            results = line.replace('\n','').split(', ')[3:]
            l.append([float(re.split(':')[1]) for re in results])

    __min = []
    __mi = 0.
    for i, item in enumerate(l):
        print(item)
        _min= item[0]
        __mi+=_min

        # if i==0: continue
        if i%_step==_step-1:
            __min.append(__mi/float(_step))
            __mi=0.

        if i == _renge: break
    mi = np.array(__min)
    
    return mi

def log_show2(filename, renge=100, step=5):
    mi = log_load2(filename, renge, step)
    ge = np.linspace(step, renge, mi.size)
    # Figure instance
    fig = plt.Figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(ge, mi, label='score')
    ax1.set_title(f'{filename}')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Generation')
    ax1.legend()

    def _destroyWindow():
        fig.savefig(f'{filename}.png')
        root.quit()
        root.destroy()



    # Tkinter Class

    root = tk.Tk()
    root.withdraw()
    root.protocol('WM_DELETE_WINDOW', _destroyWindow)  # When you close the tkinter window.

    # Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)  # Generate canvas instance, Embedding fig in root
    canvas.draw()
    canvas.get_tk_widget().pack()
    #canvas._tkcanvas.pack()

    # root
    root.update()
    root.deiconify()
    root.mainloop()