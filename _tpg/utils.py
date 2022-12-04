from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
import numpy as np
import random
import logging
import inspect
import time
import os
import re

def log_load(_title , _task, _renge, _step=5):
    l =[]

    with open(f"{_title}.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            results = line.replace('\n','').split(', ')[2:]
            # breakpoint(results)
            if len(results)<2: continue
            if f'{_task}' in results[0] and 'generation:' in results[1]:
                l.append([float(re.split(':')[1]) for re in results[2:]])
                end = int(results[1].split(':')[1])

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
    
    return mi, ma, av, end

def log_show(_title, _task, renge=100, step=5):
    mi, ma, av, end = log_load(_title, _task, renge, step)
    ends = (end//step)
    ge = np.arange((ends-mi.size)*step, ends*step, step)
    # Figure instance
    fig = plt.Figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(ge, mi, label='min')
    ax1.plot(ge, ma, label='max')
    ax1.plot(ge, av, label='ave')
    # ax1.set_title(f'{filename}')
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
        root.quit()
        root.destroy()



    # Tkinter Class
    task = f'{_task}'.replace('/', '-')
    fig.savefig(f'{_title}-{task}.png')
    # while True:
    #     try:
    #         break
    #     except FileNotFoundError:
    #         os.makedirs(f'log/{self.dir}{self.today}/')

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
    
    return mi, ma, av

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
    return mi

def log_show3(func):
    def wrapper(**kwargs):
        # for file in kwargs['filenames']:
        #     # mi, ma, av = log_load3(file, _task='any', renge=renge, step=step)
        mi, ma, av, end = func(**kwargs)

        ends = (end//kwargs['step'])
        ge = np.arange((ends-mi.size)*kwargs['step'], ends*kwargs['step'], kwargs['step'])
        # Figure instance
        fig = plt.Figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(ge, mi, label='min')
        ax1.plot(ge, ma, label='max')
        ax1.plot(ge, av, label='ave')
        # ax1.set_title(f'{filename}')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Generation')
        ax1.legend()

        def _destroyWindow():
            root.quit()
            root.destroy()



        # Tkinter Class
        # task = f'{}'.replace('/', '-')
        fig.savefig(f'{kwargs["_title"]}-chaos.png')
        # while True:
        #     try:
        #         break
        #     except FileNotFoundError:
        #         os.makedirs(f'log/{self.dir}{self.today}/')

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
        # root.mainloop()
    
    return wrapper

def log_show4(func):
    def wrapper(*args, **kwargs):
        # for file in kwargs['filenames']:
        #     # mi, ma, av = log_load3(file, _task='any', renge=renge, step=step)
        mi, ma, av, end = func(*args, **kwargs)

        ends = (end//kwargs['step'])
        ge = np.arange((ends-mi.size)*kwargs['step'], ends*kwargs['step'], kwargs['step'])
        # Figure instance
        fig = plt.Figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(ge, mi, label='min')
        ax1.plot(ge, ma, label='max')
        ax1.plot(ge, av, label='ave')
        # ax1.set_title(f'{filename}')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Generation')
        ax1.legend()

        def _destroyWindow():
            root.quit()
            root.destroy()



        # Tkinter Class
        task = f'{args[0]}'.replace('/', '-')
        fig.savefig(f'{args[0]}-{task}.png')
        # while True:
        #     try:
        #         break
        #     except FileNotFoundError:
        #         os.makedirs(f'log/{self.dir}{self.today}/')

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
    
    return wrapper

@log_show3
def log_load3(_title, _task, renge=100, step=5):
    l =[]

    with open(f"{_title}.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            results = line.replace('\n','').split(', ')[2:]
            # breakpoint(results)
            if len(results)<2: continue
            if (f'{_task}' in results[0] or f'{_task}' == 'any') and 'generation:' in results[1]:
                l.append([float(re.split(':')[1]) for re in results[2:]])
                end = int(results[1].split(':')[1])

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
        if i%step==step-1:
            __min.append(__mi/float(step))
            __mi=0.
            __max.append(__ma/float(step))
            __ma=0.
            __ave.append(__av/float(step))
            __av=0.
        if i == renge: break
    mi = np.array(__min)
    ma = np.array(__max)
    av = np.array(__ave)
    
    return mi, ma, av, end

@log_show3
def log_accumlation(_title, _task, renge=100, step=5):
    l =[]

    with open(f"{_title}.log", "r") as file:
        lines = file.readlines()
        for line in lines:
            results = line.replace('\n','').split(', ')[2:]
            # breakpoint(results)
            if len(results)<2: continue
            if (f'{_task}' in results[0] or f'{_task}' == 'any') and 'generation:' in results[1]:
                l.append([float(re.split(':')[1]) for re in results[2:]])
                end = int(results[1].split(':')[1])

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
        if i%step==step-1:
            __mi/=float(step)
            __mi+=__min[-1]
            __min.append(__mi)
            __mi=0.
            __ma/=float(step)
            __ma+=__max[-1]
            __max.append(__ma)
            __ma=0.
            __av/=float(step)
            __av+=__ave[-1]
            __ave.append(__av)
            __av=0.
        if i == renge: break

    mi = np.array(__min)
    ma = np.array(__max)
    av = np.array(__ave)
    
    return mi, ma, av, end

def flip(prob):
    """
    Coin flips, at varying levels of success based on prob.
    """
    return random.uniform(0.0,1.0) < prob
    
def breakpoint(*_print):
    import sys
    print(_print)
    sys.exit()

def getTeams(team, rec=True, visited=None, result=None):
    """
    Returns the teams that this team references, either immediate or
    recursively.
    """
    if rec:
        # recursively search all teams
        nTeams = 0

        # track visited teams to not repeat
        if visited is None:
            visited = set()
            result = list()

        visited.add(str(team.id))
        if team not in result:
            result.append(team)

        # get team count from each learner that has a team
        for lrnr in team.learners:
            lrnrTeam = lrnr.getActionTeam()
            if lrnrTeam is not None and str(lrnrTeam.id) not in visited:
                getTeams(lrnrTeam, rec=True, visited=visited, result=result)

        if len(visited) != len(result):
            print("[getTeams]Visited {} teams but got {} teans. Something is a miss!".format(len(visited), len(result)))

            print("[getTeams]visited team ids:")
            for cursor in visited:
                print(cursor)

            print("[getTeams]result team id's")
            for cursor in result:
                print(cursor.id)

        return result

    else:
        # just the teams attached directly to this team
        return [lrnr.getActionTeam() for lrnr in team.learners
            if not lrnr.isActionAtomic()]

def getLearners(team, rec=True, tVisited=None, lVisited=None, result=None, map=None):
    """
    Returns the learners on this team, immediately or recursively.
    """
    if rec:

        # track visited learners/teams to not repeat
        if tVisited is None:
            tVisited = set()
            lVisited = set()
            result = []
            map = {}

        tVisited.add(str(team.id))
        [lVisited.add(str(lrnr.id)) for lrnr in team.learners]
        
        for cursor in team.learners:
            if str(team.id) not in map:
                    map[str(team.id)] = [str(cursor.id)]
            else:
                map[str(team.id)].append(str(cursor.id))

            if cursor not in result:
                result.append(cursor)

                
        # get learner count from each learner that has a team
        for lrnr in team.learners:
            lrnrTeam = lrnr.getActionTeam()
            if lrnrTeam is not None and str(lrnrTeam.id) not in tVisited:
                getLearners(lrnrTeam, rec=True, tVisited=tVisited, lVisited=lVisited, result=result, map=map)

        if len(lVisited) != len(result):
            print("[getLearners]Visited {} learners but got {} learners. Something is a miss!".format(len(lVisited), len(result)))
            
            print("[getLearners]visited learner ids:")
            for cursor in lVisited:
                print(cursor)

            print("[getLearners]result learner id's")
            freq = {}
            for cursor in result:
                if str(cursor.id) not in freq:
                    freq[str(cursor.id)] = 1
                else:
                    freq[str(cursor.id)] = freq[str(cursor.id)] + 1
    
            print(freq)

            for cursor in freq.items():
                if cursor[1] > 1:
                    first = None
                    second = None
                    for j in result:
                        if str(j.id) == cursor[0]:
                            if first == None:
                                first = j
                            else:
                                second = j
                                break 
                    
                    print("first == second? {}".format(first.debugEq(second)))
                    print("id appears in the following teams: ")
                    for entry in map.items():
                        if str(first.id) in entry[1]:
                            print(entry[0])

        return result

    else:
        # just the teams attached directly to this team
        return list(team.learners)

def learnerInstructionStats(learners, operations):
    """
    Returns a dictionary containing counts of each type of instruction and other basic
    stats relating to instructions.
    "learners" is a list of learners that you want the stats from. "operations" is a
    list of strings representing the current operation set, can be obtained from Program.
    """

    # stats tracked for each operation and overall
    partialStats = {
        "total": 0,
        "min": float("inf"),
        "max": 0,
        "avg": 0
    }

    # dictionary that we put results in and return
    results = {"overall": partialStats.copy()}
    for op in operations:
        results[op] = partialStats.copy()

    # get instruction data from all provided learners
    for lrnr in learners:
        insts = lrnr.program.instructions
        results["overall"]["total"] += len(insts)
        results["overall"]["min"] = min(len(insts), results["overall"]["min"])
        results["overall"]["max"] = max(len(insts), results["overall"]["max"])
        results["overall"]["avg"] += len(insts)/len(learners)

        for i, op in enumerate(operations):
            opCount = np.count_nonzero(insts[:,1]==i)
            results[op]["total"] += opCount
            results[op]["min"] = min(opCount, results[op]["min"])
            results[op]["max"] = max(opCount, results[op]["max"])
            results[op]["avg"] += opCount/len(learners)

    return results

def actionInstructionStats(learners, operations):
    """
    Returns a dictionary containing counts of each type of instruction and other basic
    stats relating to instructions in action programs.
    "learners" is a list of learners that you want the stats from. "operations" is a
    list of strings representing the current operation set, can be obtained from Program.
    """

    # stats tracked for each operation and overall
    partialStats = {
        "total": 0,
        "min": float("inf"),
        "max": 0,
        "avg": 0
    }

    # dictionary that we put results in and return
    results = {"overall": partialStats.copy()}
    for op in operations:
        results[op] = partialStats.copy()

    results["numActPrograms"] = 0

    # get instruction data from all provided real atomic action learners
    for lrnr in learners:
        if not lrnr.isActionAtomic() or lrnr.actionObj.actionLength == 0:
            continue
        
        insts = lrnr.actionObj.program.instructions
        
        results["overall"]["total"] += len(insts)
        results["overall"]["min"] = min(len(insts), results["overall"]["min"])
        results["overall"]["max"] = max(len(insts), results["overall"]["max"])
        results["overall"]["avg"] += len(insts)/len(learners)

        for i, op in enumerate(operations):
            opCount = np.count_nonzero(insts[:,1]==i)
            results[op]["total"] += opCount
            results[op]["min"] = min(opCount, results[op]["min"])
            results[op]["max"] = max(opCount, results[op]["max"])
            results[op]["avg"] += opCount/len(learners)

        results["numActPrograms"] += 1

    return results

def pathDepths(team, prevDepth=0, parents=[]):
    """
    Obtains the longest execution possible in the graph from the starting (root) team.
    """

    # depth is one deeper than the last
    myDepth = prevDepth + 1
    depths = [myDepth]

    # don't revisit this team again in this depth first recursion
    parents.append(team.id)

    # the teams to visit from the learners that have team actions
    nextTeams = [lrn.getActionTeam() for lrn in team.learners 
        if not lrn.isActionAtomic() and not lrn.getActionTeam().id in parents]

    # obtain depths from each child team
    for nTeam in nextTeams:
        depths.extend(pathDepths(nTeam, myDepth, list(parents)))

    return depths

def sigmoid(x):
    # assert isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, ), f'{x} cant opelation sigmoid'
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid2(x):
    # assert isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, ), f'{x} cant opelation sigmoid'
    return (1.0 / (1.0 + np.exp(-x)))**2

def abstract(_sequence):
    assert isinstance(_sequence, np.ndarray), f'{_sequence} should be ndarray'
    state = np.array(_sequence)
    state.fill(np.nan)
    key = np.random.choice(range(state.size), random.randint(1, state.size-1))
    state[key] = _sequence[key]
    return state



class _Logger:
    _instance=None
    _logger=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            cls._instance = True
            # cls._logger= logging.getLogger(cls.__name__)
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs) -> None:
        self.dir=''

    @classmethod
    def info(cls, *args, **kwargs):
        if cls._logger: cls._logger.info(*args, extra={'className': cls.__name__ }, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        if cls._logger: cls._logger.debug(*args, extra={'className': cls.__name__ }, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        if cls._logger: cls._logger.warning(*args, extra={'className': cls.__name__ }, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        if cls._logger: cls._logger.error(*args, extra={'className': cls.__name__ }, **kwargs)

    @classmethod
    def critical(cls, *args, **kwargs):
        if cls._logger: cls._logger.critical(*args, extra={'className': cls.__name__ }, **kwargs)
    
    @classmethod
    def log(cls, *args, **kwargs):
        if cls._logger: cls._logger.log(*args, extra={'className': cls.__name__ }, **kwargs)

    @classmethod
    def set_logger(cls, _logger):
        if cls._logger is None :
            cls._logger = _logger
            for clsObj in [cls.__dict__[i] for i in cls.__dict__.keys() if inspect.isclass(cls.__dict__[i]) and re.match(r'^[A-Z]', i)]:
                clsObj.set_logger(_logger)

    @classmethod
    def unset_logger(cls):
        if cls._logger :
            for handler in cls._logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                cls._logger.removeHandler(handler)

            cls._logger=None
            for clsObj in [cls.__dict__[i] for i in cls.__dict__.keys() if inspect.isclass(cls.__dict__[i]) and re.match(r'^[A-Z]', i)]:
                clsObj.unset_logger()
            

    @property
    def task(self):
        return ''

    def add_handrer(self, _logfile, _test):
        print(f'{_logfile} {self.filename}')
        time.sleep(0.1)
        # create file handler which logs even DEBUG messages
        if not _test:
            while True:
                try:
                    _fh = logging.FileHandler(f'log/{self.dir}{self.filename}.log')
                    break
                except FileNotFoundError:
                    os.makedirs(f'log/{_logfile}')

            _fh.setLevel(logging.INFO)
            _fh_formatter = logging.Formatter('%(asctime)s, %(className)s, %(message)s')
            _fh.setFormatter(_fh_formatter)
            self.__class__._logger.addHandler(_fh)
        

        # create console handler with a INFO log level
        

        return self.filename
    
    def remove_handrer(self):
        if self.__class__._logger:
            for handler in self.__class__._logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    self.__class__._logger.removeHandler(handler)
        


    def setup_logger(self, _name, test=False, load=True):

        # self.remove_handrer()
        if self.__class__._logger is None :
            self.filename = datetime.today().strftime('%H-%M-%S')
            self.today = datetime.today().strftime('%Y-%m-%d')
            _logger = logging.getLogger(_name)
            _logger.setLevel(logging.DEBUG)
            self.set_logger(_logger)
            if load:
                _ch = logging.StreamHandler()
                _ch.setLevel(logging.DEBUG)
                _ch_formatter = logging.Formatter('[{}.%(asctime)s][%(className)s], %(message)s'.format(self.filename))
                _ch.setFormatter(_ch_formatter)

                # add the handlers to the logger
                self.__class__._logger.addHandler(_ch)
            
            # self.add_handrer(_logfile,test)
            while True:
                try:
                    _fh = logging.FileHandler(f'log/{self.dir}{self.today}/{self.filename}.log')
                    break
                except FileNotFoundError:
                    os.makedirs(f'log/{self.dir}{self.today}/')

            _fh.setLevel(logging.DEBUG)
            _fh_formatter = logging.Formatter('%(asctime)s, %(className)s, %(message)s')
            _fh.setFormatter(_fh_formatter)
            self.__class__._logger.addHandler(_fh)

        return self.__class__._logger, self.filename

    def set_level(self, _level):
        if self.__class__._logger:
            for handler in self.__class__._logger.handlers:
                handler.setLevel(_level)

    def log_load(self, _renge, _step=5):
        l =[]

        with open(f"log/{self.dir}{self.today}/{self.filename}.log", "r") as file:
            lines = file.readlines()
            for line in lines:
                results = line.replace('\n','').split(', ')[2:]
                # breakpoint(results)
                if len(results)<2: continue
                if f'{self.task}' in results[0] and 'generation:' in results[1]:
                    l.append([float(re.split(':')[1]) for re in results[2:]])
                    end = int(results[1].split(':')[1])

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
        
        return mi, ma, av, end

    def log_show(self, renge=100, step=5):
        mi, ma, av, end = self.log_load(renge, step)
        ends = (end//step)
        ge = np.arange((ends-mi.size)*step, ends*step, step)
        # Figure instance
        fig = plt.Figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(ge, mi, label='min')
        ax1.plot(ge, ma, label='max')
        ax1.plot(ge, av, label='ave')
        # ax1.set_title(f'{filename}')
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
            root.quit()
            root.destroy()



        # Tkinter Class
        task = f'{self.task}'.replace('/', '-')
        fig.savefig(f'log/{self.dir}{self.today}/{self.filename}-{task}.png')
        # while True:
        #     try:
        #         break
        #     except FileNotFoundError:
        #         os.makedirs(f'log/{self.dir}{self.today}/')

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
        # root.mainloop()
        
        return mi, ma, av

    # @log_show3
    def log_load1(self, _renge=100, _step=5):
        l =[]

        with open(f"log/{self.dir}{self.today}/{self.filename}.log", "r") as file:
            lines = file.readlines()
            for line in lines:
                results = line.replace('\n','').split(', ')[2:]
                if len(results)<2: continue
                if 'task:' in results[0] and 'generation:' in results[1]:
                    # breakpoint(results)
                    l.append([float(re.split(':')[1]) for re in results[2:]])
                    end = int(results[1].split(':')[1])

        __mi = 0.
        __ma = 0.
        __av = 0.
        __min = [0.]
        __max = [0.]
        __ave = [0.]
        for i, item in enumerate(l):
            _min, _max, _ave = item
            __mi+=_min
            __ma+=_max
            __av+=_ave
            # if i==0: continue
            if i%_step==_step-1:
                self.debug(f'end:{i}')
                __mi/=float(_step)
                __mi+=__min[-1]
                __min.append(__mi)
                __mi=0.
                __ma/=float(_step)
                __ma+=__max[-1]
                __max.append(__ma)
                __ma=0.
                __av/=float(_step)
                __av+=__ave[-1]
                __ave.append(__av)
                __av=0.
            if i == _renge: break

        mi = np.array(__min)
        ma = np.array(__max)
        av = np.array(__ave)
        
        return mi, ma, av, end

    def log_show1(self, renge=100, step=5):
        mi, ma, av, end = self.log_load1(renge, step)
        ends = (end//step)
        ge = np.arange((ends-mi.size)*step, ends*step, step)
        # Figure instance
        fig = plt.Figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(ge, mi, label='min')
        ax1.plot(ge, ma, label='max')
        ax1.plot(ge, av, label='ave')
        # ax1.set_title(f'{filename}')
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
            root.quit()
            root.destroy()



        # Tkinter Class
        # task = f'{self.task}'.replace('/', '-')
        fig.savefig(f'log/{self.dir}{self.today}/{self.filename}-chaos.png')
        # while True:
        #     try:
        #         break
        #     except FileNotFoundError:
        #         os.makedirs(f'log/{self.dir}{self.today}/')

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
        # root.mainloop()
        
        return mi, ma, av
