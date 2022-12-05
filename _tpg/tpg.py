from datetime import datetime
from math import tanh
from _tpg.utils import _Logger
from _tpg.utils import *
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gym.spaces import *

import matplotlib.pyplot as plt
import multiprocessing as mp
import tkinter as tk
import cv2
import numpy as np
import sys
import gym
import signal
import time

class _TPG(_Logger):
    Trainer=None
    taskName=None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            from _tpg.trainer import _Trainer

            cls._instance = True
            cls.Trainer = _Trainer
            # cls.Trainer._logger=cls._logger

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
            actions=None,
            teamPopSize:int=10,               # *
            rootBasedPop:bool=True,             
            gap:float=0.5,                      
            inputSize:int=33600,                
            nRegisters:int=8,                   # *
            initMaxTeamSize:int=10,             # *
            initMaxProgSize:int=10,             # *
            maxTeamSize:int=-1,                 # *
            pLrnDel:float=0.7,                  # *
            pLrnAdd:float=0.6,                  # *
            pLrnMut:float=0.2,                  # *
            pProgMut:float=0.1,                 # *
            pActMut:float=0.1,                  # *
            pActAtom:float=0.95,                # *
            pInstDel:float=0.5,                 # *
            pInstAdd:float=0.4,                 # *
            pInstSwp:float=0.2,                 # *
            pInstMut:float=1.0,                 # *
            doElites:bool=True, 
            memMatrixShape:tuple=(100,8),       # *
            rampancy:tuple=(0,0,0),
            prevPops=None, mutatePrevs=True,
            initMaxActProgSize:int=6,           # *
            nActRegisters:int=4,
        ):
        self.trainer = self.__class__.Trainer(
            actions=actions, 
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,         # *
            initMaxProgSize=initMaxProgSize,         # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pActMut=pActMut,                  # *
            pActAtom=pActAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nActRegisters=nActRegisters)
        
        self.generations=10
        self.episodes=1
        self.frames = 500
        self.show = False
        self.test = False
        self.load = True
        self.dir = 'test/'

        self.env = None
        self.tasks = set()
        self.scores = {}
        self.gen=0

    def setActions(self, actions):
        self.actions = self.trainer.setActions(actions)

    def setEnv(self, env):
        self.env = env
        self.tasks.add(self.task)
        self.state = self.env.reset()
        self.setActions(self.env.action_space.n)
        self.__class__.taskName=self.task

    def getAgents(self):
        return self.trainer.getAgents()
    
    def setAgents(self, task=None):
        self.scores = {}
        self.agents = self.trainer.getAgents(task = task if task else self.task)

    def flush_render(self, step=0, name='', info=''):
        self.ax1.imshow(self.env.render(mode='rgb_array'))
        self.ax1.set_title(f'{name}| Step: {step} {info}')
        self.ax1.axis('off')

    def set_tk_render(self):
        # ge = np.linspace(step, renge, mi.size)
        # Figure instance
        self.fig = plt.Figure()

        self.ax1 = self.fig.add_subplot(111)

        def _destroyWindow():
            self.root.quit()
            self.root.destroy()

        # Tkinter Class

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.protocol('WM_DELETE_WINDOW', _destroyWindow)  # When you close the tkinter window.

        # Canvas
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # Generate canvas instance, Embedding fig in root
        canvas.draw()
        canvas.get_tk_widget().pack()
        #canvas._tkcanvas.pack()

        # root
        self.root.update()
        self.root.deiconify()
        self.root.mainloop()

    def getState(self,inState):
        # each row is all 1 color
        rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

        # add each with appropriate shifting
        # get RRRRRRRR GGGGGGGG BBBBBBBB
        return np.add(np.left_shift(rgbRows[0], 16),
            np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))

    def episode(self):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
        
        for agent in self.agents: # to multi-proccess
            state = self.env.reset()
            score = 0
            for i in range(self.frames): # run episodes that last 500 frames
                act = agent.act(state)
                if not act in range(self.env.action_space.n): continue
                state, reward, isDone, debug = self.env.step(act)
                score += reward # accumulate reward in score
                # self.info(f'state:{state}')


                if isDone: break # end early if losing state

            agent.score+=score

    def generation(self):
        """ section """
        # _scores = {}
        self.setAgents()
        for _ in range(self.episodes):
            self.episode()

        for agent in self.agents:               
            agent.score/=self.episodes
            agent.reward(task=self.task)
            self.scores[agent.id]=agent.score
        
        self.trainer.evolve(list(self.tasks))

        self.info(f'task:{self.task}, generation:{self.gen}, min:{min(self.scores.values())}, max:{max(self.scores.values())}, ave:{sum(self.scores.values())/len(self.scores)}')
        
        self.gen+=1
    
    def story(self, _trainer=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=None, _test=None, _load=None, _dir=None):
        self.prologue(_trainer=_trainer, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load, _dir=_dir)
        self.debug(f'story_task:{self.task}, gen:{self.gen}')

        for _ in range(self.generations): # generation loop
            self.generation()

        return self.epilogue()

    def story1(self, *args, **kwargs):
        self.prologue1(*args, **kwargs)
        self.debug(f'story_task:{self.task}, gen:{self.gen}')
        for _ in range(self.generations):
            self.generation()
        
        return self.epilogue1()

    def success_story(self, _trainer=None, _task:str=None, _generations:int=1, _episodes:int=None, _frames:int=None, _show=None, _test=None, _load=None, _dir=None):
        self.prologue(_trainer=_trainer, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load, _dir=_dir)
        self.debug(f'story_task:{self.task}, gen:{self.gen}')

        for _ in range(self.generations): # generation loop
            score=0
            for i in range(self.frames): # run episodes that last 500 frames
                act = self.elite.act(state)
                if not act in range(self.env.action_space.n): continue
                state, reward, isDone, debug = self.env.step(act)
                score += reward # accumulate reward in score
                # self.info(f'state:{state}')


                if isDone: break # end early if losing state

            self.info(f'task:{self.task}, time:{_}, elite_score:{score}')

        self.env.close()

        return f'log/{self.dir}{self.today}/{self.filename}'

    def chaos_story(self, **kwargs):
        random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))

        
        self.setEnv(gym.make(random.choice(kwargs['_tasks'])))
        self.prologue1(**kwargs)
        for _ in range(self.generations):
            self.frames=random.randint(0,1000)
            self.generation()
            self.env.close()
            self.setEnv(gym.make(random.choice(kwargs['_tasks'])))

        return self.epilogue1()

    def multi(self, _tasks, _generations=None, _load=None):
        self.archive=[]
        for task in _tasks:
            title = self.story(_task=task, _generations=_generations, _load=_load)
            self.archive+=[title]

        return self.archive

    
    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, self.__class__.Trainer): raise Exception(f'this object is not {self.__class__.Trainer}')

    def prologue(self, _trainer=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=None, _test=None, _load=None, _dir=None):
        if _trainer:
            self.instance_valid(_trainer)
            self.trainer = _trainer
        
        if _task:
            env = gym.make(_task)
            self.setEnv(env)
        
        if _test:
            self.test = _test
        if _dir:
            self.dir = _dir
        if _load:
            self.load = _load
        if _generations:
            self.generations=_generations
        if _episodes:
            self.episodes = _episodes
        if _frames:
            self.frames = _frames
        if _show:
            self.show = _show

        self.setup_logger(__name__, test=self.test, load=self.load)

        def interruption(signum, frame):
            self.epilogue()
            print('interruption')
            sys.exit()
        
        signal.signal(signal.SIGINT, interruption)

    def prologue1(self, *args, **kwargs):
        if kwargs.get('_trainer'):
            self.instance_valid(kwargs['_trainer'])
            self.trainer = kwargs['_trainer']
        
        if kwargs.get('_task'):
            env = gym.make(kwargs['_task'])
            self.setEnv(env)
        
        if kwargs.get('_test'):
            self.test = kwargs['_test']
        if kwargs.get('_dir'):
            self.dir = kwargs['_dir']
        if kwargs.get('_load'):
            self.load = kwargs['_load']
            # self.set_level(logging.DEBUG)
        if kwargs.get('_generations'):
            self.generations=kwargs['_generations']
        if kwargs.get('_episodes'):
            self.episodes = kwargs['_episodes']
        if kwargs.get('_frames'):
            self.frames = kwargs['_frames']
        if kwargs.get('_show'):
            self.show = kwargs['_show']
        self.setup_logger(__name__, test=self.test, load=self.load)
        self.debug(f'gen:{self.gen}, task:{self.task}, frames:{self.frames}')

        def interruption(signum, frame):
            self.epilogue1()
            print('interruption')
            sys.exit()
        
        signal.signal(signal.SIGINT, interruption)

    def epilogue(self):
        # title = f'{self.filename}'
        self.env.close()
        self.log_show()
        self.trainer.save(f'log/{self.dir}{self.today}/{self.filename}')

        return f'log/{self.dir}{self.today}/{self.filename}'

    def epilogue1(self, **kwargs):
        self.env.close()
        self.log_show1()
        self.trainer.save(f'log/{self.dir}{self.today}/{self.filename}')
        return f'log/{self.dir}{self.today}/{self.filename}'

    @property
    def task(self):
        return self.env.spec.id

    @property
    def elite(self):
        return self.trainer.getElite(list(self.tasks))

    # @classmethod
    def load_story(self, _title):
        _trainer = self.__class__.Trainer.load(_title)
        self.trainer = _trainer
        
class MTPG(_TPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer
            cls._instance = True
            cls.Trainer = Trainer

        return super().__new__(cls, *args, **kwargs)

class MHTPG(_TPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer1
            cls._instance = True
            cls.Trainer = Trainer1

        return super().__new__(cls, *args, **kwargs)

    # def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
    #     self.trainer.evolve(tasks, multiTaskType, extraTeams)

class ActorTPG(MHTPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer1_1
            cls._instance = True
            cls.Trainer = Trainer1_1

        return super().__new__(cls, *args, **kwargs)

    def __getitem__(self, __code):
        return self.actions[__code]

    def setAgents(self, task='task'):
        self.agents = self.trainer.getAgents(task=task)
        self.actionSequence = {}
        self.actionReward = {}
        for actor in self.agents:
            self.actionSequence[actor.id]=[]
            self.actionReward[actor.id]=0.
        
        return self.agents

    def getEliteAgent(self, task='task'):
        return self.trainer.getEliteAgent(task)

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _actionSequence=None, _actionReward=None):
        self.trainer.evolve(tasks=tasks, _actionSequence=_actionSequence, _actionReward=_actionReward)

    def episode(self):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
        
        for agent in self.agents: # to multi-proccess
            state = self.env.reset() # get initial state and prep environment
            score = 0
            frame = 0
            
            while frame<self.frames: # run episodes that last 500 frames
                acts = agent.act(state.flatten()).action
                for action in acts:
                    frame+=1
                    if not action in range(self.env.action_space.n): continue
                    state, reward, isDone, debug = self.env.step(action)
                    score += reward # accumulate reward in score
                    self.actionSequence[agent.id]+=[action]

                    if isDone: break # end early if losing state
                    if frame>self.frames: break
                    if self.show:self.flush_render()


            agent.score += score # store score

class Actor(_TPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer1_2
            
            cls._instance = True
            cls.Trainer = Trainer1_2

        return super().__new__(cls, *args, **kwargs)

    def __getitem__(self, __code):
        return self.actions[__code]

    # def __call__(self, actionSequence):
    #     return self.__class__.Trainer.ActionObject(actionSequence)
    #     # return super().__call__(*args, **kwds)

    def setAgents(self, task=None):
        self.agents = self.trainer.getAgents(task=task if task else self.task)
        self.scores = {}  
        return self.agents
    
    def getAgents(self, task=None):
        return self.trainer.getAgents(task=task if task else self.task)

    def getEliteAgent(self, task=None):
        return self.trainer.getEliteAgent(task if task else self.task)

    def activator(self, acts):
        assert isinstance(acts, np.ndarray), f'{acts} cant fetch activation'
        try:
            # if acts == (np.array([], dtype=np.float64)): raise Exception('action is Null')
            # self.info(f'acts_size:{acts.size}')
            if isinstance(self.env.action_space, Box):
                shape = self.env.action_space.shape
                acts = np.split(acts, np.prod(shape, dtype=np.int8))
                if acts.size < 1 :
                    self.frame = self.frames
                    return
                for action in acts:
                    # self.info(f'act:{action}')
                    if self.frame>self.frames: break
                    self.frame+=1
                    self.state, reward, isDone, debug = self.env.step(action.reshape(shape))
                    self.score+=reward
                    self.sequence+=[action]
                    if isDone: self.frame=self.frames
            elif isinstance(self.env.action_space, Discrete):
                acts = acts[acts<self.env.action_space.n]
                if acts.size < 1 :
                    self.frame = self.frames
                    return
                for action in acts:
                    if self.frame>self.frames: break
                    # self.info(f'act:{action}')
                    self.state, reward, isDone, debug = self.env.step(int(action))
                    self.frame+=1
                    self.score+=reward
                    self.sequence+=[action]
                    if isDone: self.frame=self.frames
            else:
                print(self.env.action_space)
                raise 'EnvExpectation'
        except Exception as e:
            breakpoint(e)


    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self.trainer.evolve(tasks=tasks)

    def episode(self):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
        
        for agent in self.agents: # to multi-proccess
            self.state = self.env.reset() # get initial state and prep environment
            self.score = 0
            self.frame = 0
            self.sequence = []
            while self.frame<self.frames: # run episodes that last 500 frames
                acts = agent.act(self.state.flatten()).action

                self.activator(acts)

            agent.trace(self.sequence)
            agent.score+=self.score

    def generation(self):
        self.setAgents()
        for _ in range(self.episodes):     
            self.episode()

        for agent in self.agents:
            agent.score/=self.episodes
            self.scores[agent.id]=agent.score
            agent.reward(task=self.task)
        
        self.evolve(list(self.tasks))
        self.info(f'task:{self.task}, generation:{self.gen}, min:{min(self.scores.values())}, max:{max(self.scores.values())}, ave:{sum(self.scores.values())/len(self.scores)}')
        # self.info(f'actions:{repr(self.actions)}')
        self.gen+=1

    def story(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        self.prologue(_trainer=_trainer, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load, _dir=_dir)


        for _ in range(self.generations): # generation loop
            self.generation()

        
        return self.epilogue()

class Actor1(Actor):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer1_2_1
            cls._instance = True
            cls.Trainer = Trainer1_2_1

        return super().__new__(cls, *args, **kwargs)

class Actor2(Actor):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer1_2_2
            cls._instance = True
            cls.Trainer = Trainer1_2_2

        return super().__new__(cls, *args, **kwargs)

class Opelator(Actor):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer3
            cls._instance = True
            cls.Trainer = Trainer3

        return super().__new__(cls, *args, **kwargs)

class EmulatorTPG(_TPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer2
            cls._instance = True
            cls.Trainer = Trainer2

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        state=None,
        teamPopSize:int=10,                 # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pMemMut:float=0.1,                  # *
        pMemAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nMemRegisters:int=4,
    ):
        self.trainer = self.__class__.Trainer(
            state=state, 
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pMemMut=pMemMut,                  # *
            pMemAtom=pMemAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nMemRegisters=nMemRegisters)
        
        self.generations=10
        self.episodes=1
        self.frames = 500
        self.show = False
        self.test = False
        self.dir = ''

    def __getitem__(self, __code):
        return self.memories[__code]

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _states=None, _rewards=None, _unexpectancies=None):
        self.trainer.evolve(tasks, multiTaskType, extraTeams, _states, _rewards, _unexpectancies)
    def setEnv(self, env):
        self.env = env
        self.tasks.add(self.env.spec.id)
        self.state = self.env.reset()
        self.setMemories(self.state)

    def setMemories(self, state):
        self.memories = self.trainer.setMemories(state.flatten())

    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
      
        states = []
        unexpectancies = []
        for agent in self.agents: # to multi-proccess
            
            state = self.env.reset() # get initial state and prep environment
            score = 0
            for _ in range(self.frames): # run episodes that last 500 frames
                act = self.env.action_space.sample()
                imageCode = agent.image(act, state.flatten())
                state, reward, isDone, debug = self.env.step(act)
                diff, unex = self.memories[imageCode].memorize(state.flatten(), reward)

                score += tanh(np.power(diff, 2).sum())
                states+=[state.flatten()]
                unexpectancies+=[unex]

                if isDone: break # end early if losing state
                # if self.show: self.show_state(self.env, _)

            # if _scores.get(_id) is None : _scores[_id]=0
            # _scores[_id] += score # store score
            agent.score+=score
            if self.__class__.logger is not None: self.debug(f'{agent.id}:{score}')

        return _scores, states, unexpectancies

    def generation(self):
        _scores = {}
        self.setAgents()
        # _task = self.env.spec.id
        for _ in range(self.episodes):  _scores, states, unexpectancies = self.episode(_scores)
        # for i in _scores:               _scores[i]/=self.episodes
        for agent in self.agents:
            agent.score/=self.episodes
            _scores[agent.id]=agent.score
            agent.reward(task=self.task)
        self.trainer.evolve([self.task], _states=states, _unexpectancies=unexpectancies)

        return _scores
   
    def story(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        self.prologue(_trainer=_trainer, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load, _dir=_dir)

        for gen in tqdm(range(self.generations)): # generation loop
            scores = self.generation()
            self.info(f'generation:{gen}, min:{min(scores.values())}, max:{max(scores.values())}, ave:{sum(scores.values())/len(scores)}')

        self.epilogue()

        return f'{self.dir+self.task}/{self.filename}'

    def prologue(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        if _trainer: 
            self.instance_valid(_trainer)
            self.trainer = _trainer
        
        if _task:
            self.env = gym.make(_task)
        
        if _test:
            self.test = _test
        if _dir:
            self.dir = _dir
        # task = _dir+self.env.spec.id

        logger, self.filename = setup_logger(__name__, f'{self.dir+self.task}', test=_test, load=_load)
        self.__class__.logger = logger
        self.generations = _generations
        self.episodes = _episodes
        self.frames = _frames
        self.show = _show

        self.setMemories(state=self.state.flatten())

        def interruption(signum, frame):
            self.epilogue()
            print('interruption')
            sys.exit()
        
        signal.signal(signal.SIGINT, interruption)


    def epilogue(self):
        title = f'{self.dir+self.task}/{self.filename}'
        map(self.__class__.logger.removeHandler, self.__class__.logger.handlers)
        map(self.__class__.logger.removeFilter, self.__class__.logger.filters)
        if not self.test: self.trainer.save(title)
        log_show(title)

class EmulatorTPG1(EmulatorTPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer2_1
            cls._instance = True
            cls.Trainer = Trainer2_1
            cls.Trainer._logger=cls._logger

        return super().__new__(cls, *args, **kwargs)

    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
      
        states = []
        unexpectancies = []
        print('agents loop')
        for agent in tqdm(self.agents): # to multi-proccess
            
            state = self.env.reset() # get initial state and prep environment
            score = 0
            _id = str(agent.team.id)
            for _ in range(self.frames): # run episodes that last 500 frames
                act = self.env.action_space.sample()
                img = agent.image(act, state.flatten()).memory
                state, reward, isDone, debug = self.env.step(act)
                diff, unex = img.compare(state.flatten(), reward)

                # オーバーフロー防止のためアークタンジェントに二乗和誤差を入れている。
                score += tanh(np.power(diff, 2).sum())
                states+=[state.flatten()]
                unexpectancies+=[unex]

                if isDone: break # end early if losing state
                if self.show: self.show_state(self.env, _)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

            # if self.logger is not None: self.logger.info(f'{_id},{score}')

        return _scores, states, unexpectancies

class Emulator(EmulatorTPG1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer2_2
            cls._instance = True
            cls.Trainer = Trainer2_2

        return super().__new__(cls, *args, **kwargs)

    def __call__(self, memorySequence):
        return self.__class__.Trainer.MemoryObject(memorySequence)

    def getAgents(self, task=None):
        return self.trainer.getAgents(task=task if task else self.task)

    def getEliteAgent(self, task=None):
        return self.trainer.getEliteAgent(task if task else self.task)

    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
      
        states = []
        unexpectancies = []
        # print('agents loop')
        for agent in self.agents: # to multi-proccess
            
            state = self.env.reset() # get initial state and prep environment
            score = 0
            _id = agent.id
            for _ in range(self.frames): # run episodes that last 500 frames
                act = self.env.action_space.sample()
                img = agent.image(act, state.flatten()).memory
                state, reward, isDone, debug = self.env.step(act)
                diff, unex = img.compare(state.flatten(), reward)

                # オーバーフロー防止のためアークタンジェントに二乗和誤差を入れている。
                score += tanh(diff)
                states+=[state.flatten()]
                unexpectancies+=[unex]

                if isDone: break # end early if losing state
                if self.show: self.show_state(self.env, _)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

            # if self.logger is not None: self.logger.info(f'{_id},{score}')

        return _scores, states, unexpectancies

class Emulator1(Emulator):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer2_3
            cls._instance = True
            cls.Trainer = Trainer2_3

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        primitive=None,
        teamPopSize:int=10,                 # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pMemMut:float=0.1,                  # *
        pMemAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nMemRegisters:int=4,
    ):
        self.trainer = self.__class__.Trainer(
            primitive=primitive, 
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pMemMut=pMemMut,                  # *
            pMemAtom=pMemAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nMemRegisters=nMemRegisters)
        
        self.generations=10
        self.episodes=1
        self.frames = 500
        self.show = False
        self.test = False
        self.dir = ''
        self.tasks = set()

    def evolve(self, multiTaskType='min', extraTeams=None):
        self.trainer._scoreIndividuals(
            list(self.tasks), 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self.trainer._saveFitnessStats() # save fitness stats
        self.trainer._select(extraTeams) # select individuals to keep
        self.trainer._generate(extraTeams) # create new individuals from those kept
        self.trainer._nextEpoch()

    def episode(self):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
      
        states = []
        for agent in self.agents: # to multi-proccess
            
            state = self.env.reset() # get initial state and prep environment
            score = 0
            for _ in range(self.frames): # run episodes that last 500 frames
                act = self.env.action_space.sample()
                image = np.array([act]+list(state.flatten()))
                img = agent.image(image).memory
                state, reward, isDone, debug = self.env.step(act)
                diff = img.compare(image)

                # オーバーフロー防止のためアークタンジェントに二乗和誤差を入れている。
                score += tanh(diff)
                states+=[image]

                if isDone: break # end early if losing state
                # if self.show: self.show_state(self.env, _)

            # if _scores.get(_id) is None : _scores[_id]=0
            # _scores[_id] += score # store score
            agent.score+=score
            agent.trace(states)

            # if self.logger is not None: self.logger.info(f'{_id},{score}')

        return 

    def generation(self):
        self.scores = {}
        self.setAgents()
        # _task = self.env.spec.id
        for _ in range(self.episodes): self.episode()
        # for i in _scores:               _scores[i]/=self.episodes
        for agent in self.agents:
            agent.score/=self.episodes
            self.scores[agent.id]=agent.score
            agent.reward(task=self.task)

        self.evolve()

        return self.scores
   
    def story(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        self.prologue(_trainer=_trainer, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load, _dir=_dir)

        for gen in tqdm(range(self.generations)): # generation loop
            scores = self.generation()
            self.info(f'generation:{gen}, min:{min(scores.values())}, max:{max(scores.values())}, ave:{sum(scores.values())/len(scores)}')

        self.epilogue()

        return f'{self.dir+self.task}/{self.filename}'

class EmulatorEye(Emulator):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer2_3
            cls._instance = True
            cls.Trainer = Trainer2_3

            cls.Trainer._logger=cls._logger

        return super().__new__(cls, *args, **kwargs)

    def setEnv(self, env):
        self.env = env
        self.task = 'camera-memorize'
        self.state = self.env.reset()

    def setMemories(self, state):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            _, state = self.cap.read()
            self.memories = self.trainer.setMemories(state.flatten())
        # cap.release()

    def episode(self, _scores):
        states = []
        unexpectancies = []
        # cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            score = 0

            for agent in tqdm(self.agents):
                for i in range(self.frames):
                    _, state = self.cap.read()
                    if not _: continue
                    img = agent.image(np.nan, state.flatten()).memory
                    diff = img.compare(state.flatten())
                    score += tanh(diff)
                    states+=[state.flatten()]

                agent.score+=score
                agent.trace(states)

        else: breakpoint('video is not open')
        
        return _scores, states, unexpectancies

    def epilogue(self):
        if not self.test: self.trainer.save(f'{self.dir+self.task}/{self.filename}')
        map(self.__class__.logger.removeHandler, self.__class__.logger.handlers)
        map(self.__class__.logger.removeFilter, self.__class__.logger.filters)
        self.cap.release()

class _Automata(_TPG):
    Actor=None      # 運動野
    Emulator=None   # 感覚野
    _instance=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            cls._instance = True
            cls.Actor = MHTPG
            cls.Emulator = EmulatorTPG
        return super().__new__(cls)

    def __init__(self, 
        actions=None,
        states=None,
        teamPopSize:int=10,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pActMut:float=0.1,                  # *
        pActAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4,
        thinkingTime=7,
    ):
        # super().__init__()
        self.actor = self.__class__.Actor(
            actions=actions,
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pActMut=pActMut,                  # *
            pActAtom=pActAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nActRegisters=nActRegisters
        )
        self.emulator = self.__class__.Emulator(
            state=states,
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pMemMut=pActMut,                  # *
            pMemAtom=pActAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nMemRegisters=nActRegisters
        )
        self.thinkingTimeLimit=thinkingTime

        self.generations=10
        self.episodes=1
        self.frames = 500
        self.show = False
        self.test = False
        self.load = True
        self.dir = 'test/'

        self.env = None
        self.tasks = set()
        self.scores = {}
        self.gen=0
        self.initParams()

    def _setupParams(self, actor_id, emulator_id, actions=None, images=None, rewards=None):
        self.actions[actor_id]=[]
        # self.predicted_reward[actor_id]=0.
        if not self.predicted_reward.get(actor_id) : self.predicted_reward[actor_id]=0.
        if rewards is not None: self.predicted_reward[actor_id] = sum(rewards)
        if actions is not None: self.actions[actor_id] = actions

        self.pairs[actor_id] = emulator_id
        if not self.memories.get(emulator_id) : self.memories[emulator_id]=[]
        if images is not None: self.memories[emulator_id]   += images

    def _set_env_action_space(self):
        return list(range(self.env.action_space.n))+[np.nan]

    def setAction(self, action):
        self.actor.setActions(action)

    def setMemory(self, state):
        self.emulator.setMemories(state.flatten())

    def setAgents(self):
        """set hippocampus"""
        self.actor_scores = {}
        self.emulator_scores = {}
        self.actual_states = []
        self.actual_rewards = []
        self.unexpectancy = []
        self.actors = self.actor.getAgents()
        self.emulators = self.emulator.getAgents()
        for actor in self.actors:
            self.actor_scores[actor.id] = 0.
            assert not self.actor_scores.get(actor.id), f'{actor.id} cant assaign {self.actor_scores}'
        for emulator in self.emulators:
            self.emulator_scores[emulator.id] = 0.
            assert not self.emulator_scores.get(emulator.id)

    def instance_valid(self, _actor=None, _emulator=None) -> None:
        assert self.__class__.Actor.Trainer is not self.__class__.Emulator.Trainer

        if _actor: 
            assert isinstance(_actor, self.__class__.Actor.Trainer), f'this actor is not {self.__class__.Actor}'
        if _emulator:
            assert isinstance(_emulator, self.__class__.Emulator.Trainer), f'this emulator is not {self.__class__.Emulator}'

    def initParams(self):
        self.actions    = {}
        self.memories   = {}
        self.pairs      = {}
        self.predicted_reward    = {}

    def unconsiousness(self, state):
        return self.actor.elite(state)

    def think(self, cerebral_cortex, _actor, _emulator):
        self.__class__.Emulator.Trainer.MemoryObject.memories = cerebral_cortex['memories']
        self.__class__.Actor.Trainer.ActionObject.actions = cerebral_cortex['actions']
        
        assert isinstance(_actor, self.__class__.Actor.Trainer.Agent)
        assert isinstance(_emulator, self.__class__.Emulator.Trainer.Agent)
        assert self.__class__.Actor.Trainer.ActionObject.actions is not None
        assert self.__class__.Emulator.Trainer.MemoryObject.memories is not None

        state = np.array(cerebral_cortex['now'])
        
        actionCodes = []
        memoryCodes = []
        predict_rewards     = []
        timeout_start = time.time()
        # print(self.thinkingTimeLimit, _actor.id)
        for _ in range(cerebral_cortex['frames']):
            actionCode = _actor.act(state)
            imageCode  = _emulator.image(actionCode, state)
            actionCodes  += [actionCode]
            memoryCodes  += [imageCode]
            predict_rewards += [cerebral_cortex['memories'][imageCode].reward]
            state = cerebral_cortex['memories'][imageCode].recall(state)
            # breakpoint(self.thinkingTimeLimit)
        return _actor.id, _emulator.id, actionCodes, memoryCodes, predict_rewards

    def thinker(self):
        """
        意識にとって何が最善の行動となるかの選別
        """
        manager = mp.Manager()
        cerebral_cortex = manager.dict()

        # 脳皮質内の信号
        cerebral_cortex['actions']  = self.actor.actions
        cerebral_cortex['memories'] = self.emulator.memories
        # determined actionによって制限された意識チャンネル範囲内のhippocampusの情報を皮質に流す。
        cerebral_cortex['now'] = self.consciousness
        cerebral_cortex['frames'] = self.frames
        
        # 認識・計画
        with mp.Pool(mp.cpu_count()-2) as pool:
            results = pool.starmap(self.think, [(cerebral_cortex, actor, emulator) for actor, emulator in zip(self.actors, self.emulators)])
        for result in results:
            actor_id, emulator_id, actions, images, rewards = result
            self._setupParams(actor_id=actor_id, emulator_id=emulator_id, actions=actions, images=images, rewards=rewards)
            assert self.pairs.get(actor_id), (self.pairs[actor_id], emulator_id)

        # 行動の決定
        bestActor=max(self.predicted_reward, key=lambda k: self.predicted_reward.get(k))

        return bestActor, self.pairs[bestActor]

    def episode(self):
        """
        記憶単位
        行動と報酬のエピソードを生成。
        エピソードはAutomata.hippocampusに電気的記憶として保存される。
        睡眠期と行動期を分ける？
        """
        frame=0
        executor = ThreadPoolExecutor(max_workers=2)
        # state = _env.reset() # get initial state and prep environment
        thinker = executor.submit(self.thinker)
        # self.best = str(_elite_actor.team.id)
        total_reward = 0.
        # thinking_actor = []
        while frame<self.frames:
            if thinker.done(): # 意識的行動
                # thinking_actor.append(self.best)
                bestActor, pairEmulator = thinker.result()
                assert bestActor in list(self.actions.keys()), f'dont has {bestActor} in {self.actions.keys()}'
                assert pairEmulator in list(self.memories.keys()), f'dont has {pairEmulator} in {self.memories.keys()}'
                

                for actionCode, imageCode in zip(self.actions[bestActor], self.memories[pairEmulator]):
                    frame+=1
                    if not self.actor.actions[actionCode] in range(self.env.action_space.n): continue
                    state, reward, isDone, debug = self.env.step(self.actor.actions[actionCode]) # state : np.ndarray.flatten
                    if isDone: 
                        frame=self.frames
                        reward=0

                    # assert self.actor_scores.get(bestActor), f'{bestActor} not in the {self.actor_scores}'
                    if not self.actor_scores.get(bestActor): self.actor_scores[bestActor]=0.
                    self.actor_scores[bestActor]+=reward

                    # assert self.emulator_scores.get(pairEmulator), f'{pairEmulator} not in the {self.emulator_scores}'
                    if not self.emulator_scores.get(pairEmulator): self.emulator_scores[pairEmulator]=0.
                    diff, unex = self.emulator.memories[imageCode].memorize(state.flatten(), reward)
                    self.emulator_scores[pairEmulator] += tanh(np.power(diff, 2).sum())
                    self.actual_states += [state.flatten()]
                    self.actual_rewards += [reward]
                    self.unexpectancy  += [unex]
                    total_reward += reward
 
                    if isDone:
                        self.state=self.env.reset()
                        break

                    if self.show:  self.show_state(self.env, frame)
                    self.state = state
                    print(frame)

                thinker = executor.submit(self.thinker)
            else: # 無意識的行動
                """
                Automata.hippocampusに明記
                """
                # actionCode = _elite_actor.act(EmulatorTPG.state.flatten())
                # if not ActionObject3.actions[actionCode] in range(_env.action_space.n): continue

                # EmulatorTPG.state, reward, isDone, debug = _env.step(ActionObject3.actions[actionCode])
                # total_reward += reward

                # if isDone: 
                #     EmulatorTPG.state = _env.reset()
                #     break
                # if _show:  self.show_state(_env, frame)
                pass

        thinker.cancel()
        return total_reward

    def generation(self):
        _task   = self.env.spec.id
        self.setAgents()
        # breakpoint(type(emulators))
        print('start episode...')
        for _ in range(self.episodes):
            total_reward = self.episode()
        print('... end episode')

        # 報酬の贈与
        for actor in self.actors:
            actor.reward(score=self.actor_scores[actor.id], task=_task)

        # ここらへんのエミュレータの報酬設計
        for emulator in self.emulators:
            emulator.reward(score=self.emulator_scores[emulator.id], task=_task)

        self.actor.evolve([_task])
        self.emulator.evolve([_task], _states=self.actual_states, _rewards=self.actual_rewards, _unexpectancies=self.unexpectancy)

        self.initParams()
        return total_reward
    
    def story(self, _actor=None, _emulator=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=False, _test=False, _load=True, _dir=''):
        if _actor: 
            self.instance_valid(_actor)
            self.actor.trainer = _actor
        if _emulator: 
            self.instance_valid(_emulator)
            self.emulator.trainer = _emulator
        if _task:
            self.env = gym.make(_task)
        if _generations:
            self.generations=_generations
        if _episodes:
            self.episodes = _episodes
        if _frames:
            self.frames = _frames
        
        task = _dir+self.env.spec.id

        logger, filename = setup_logger(__name__, task, test=_test, load=_load)

        action_space = self.env.action_space
        observation_space = self.env.observation_space

        
        action = 2
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n

        state = observation_space.sample()
        self.setAction(action=action)
        self.setMemory(state=state.flatten())
        self.state = self.env.reset()
        def outHandler(signum, frame):
            if not _test: 
                self.actor.trainer.save(f'{task}/{filename}-act')
                self.emulator.trainer.save(f'{task}/{filename}-emu')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        summaryScores = []

        tStart = time.time()
        for gen in tqdm(range(self.generations)): # generation loop
            # breakpoint(type(_emulator))
            total_score = self.generation()
            # score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            logger.info(f'generation:{gen}, score:{total_score}', extra={'className': self.__class__.__name__})
            summaryScores.append(total_score)

        map(logger.removeHandler, logger.handlers)
        map(logger.removeFilter, logger.filters)

        return f'{task}/{filename}'
    
    @property
    def consciousness(self):
        # self.consciousness_channel_key = []
        # return self.__class__.hippocampus(self.consciousness_channel_key)
        return self.state.flatten()

class Automata1(_Automata):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            cls._instance = True
            cls.Actor = Actor
            cls.Emulator = Emulator

            cls.Actor._logger=cls._logger
            cls.Emulator._logger=cls._logger
            # cls.ActionObject = _ActionObject
            # cls.MemoryObject = _MemoryObject
        return super().__new__(cls)

    def setAgents(self):
        """set hippocampus"""
        self.actor_scores = {}
        self.emulator_scores = {}
        self.actual_states = []
        self.actual_actions = []
        self.actual_rewards = []
        self.unexpectancy = []
        self.actors = self.actor.getAgents(task=self.task)
        self.emulators = self.emulator.getAgents(task=self.task)
        self.episodes = len(self.actors)
        for actor in self.actors:
            self.actor_scores[actor.id] = 0.
            assert not self.actor_scores.get(actor.id), f'{actor.id} cant assaign {self.actor_scores}'
        for emulator in self.emulators:
            self.emulator_scores[emulator.id] = 0.
            assert not self.emulator_scores.get(emulator.id)

    def _setupParams(self, actor_id, emulator_id, actions=None, images=None, rewards=None):
        self.actions[actor_id]=[]
        # self.predicted_reward[actor_id]=0.
        if not self.predicted_reward.get(actor_id) : self.predicted_reward[actor_id]=0.
        if rewards is not None: self.predicted_reward[actor_id] = sum(rewards)
        if actions is not None: self.actions[actor_id] = self.activator(actions)

        self.pairs[actor_id] = emulator_id
        if not self.memories.get(emulator_id) : self.memories[emulator_id]=[]
        if images is not None: self.memories[emulator_id]   += images

    def think(self, cerebral_cortex, _actor, _emulator):
        self.__class__.Emulator.Trainer.MemoryObject.memories = cerebral_cortex['memories']
        self.__class__.Actor.Trainer.ActionObject.actions = cerebral_cortex['actions']
        
        assert isinstance(_actor, self.__class__.Actor.Trainer.Agent)
        assert isinstance(_emulator, self.__class__.Emulator.Trainer.Agent)
        assert self.__class__.Actor.Trainer.ActionObject.actions is not None
        assert self.__class__.Emulator.Trainer.MemoryObject.memories is not None

        state = np.array(cerebral_cortex['now'])
        
        # actions = []
        memories = []
        predict_rewards     = []
        timeout_start = time.time()
        # print(self.thinkingTimeLimit, _actor.id)
        # for _ in range(cerebral_cortex['frames']):
        act = _actor.act(state)
        # breakpoint(actions)
        actions = self.activator(act)
        i = 0
        for action in actions:
            # assert action in self.__class__.Actor.Trainer.ActionObject.actions.values(), f'{action} , {act}'
            image  = _emulator.image(action, state)
            memories  += [image]
            predict_rewards += [image.reward]
            state = state+image.state
            # breakpoint(self.thinkingTimeLimit)
            i+=1
            if i > cerebral_cortex['frames']: break
        return _actor.id, _emulator.id, act, memories, predict_rewards

    def thinker(self):
        """
        意識にとって何が最善の行動となるかの選別
        """
        manager = mp.Manager()
        cerebral_cortex = manager.dict()

        # 脳皮質内の信号
        cerebral_cortex['actions']  = self.actor.actions
        cerebral_cortex['memories'] = self.emulator.memories
        # determined actionによって制限された意識チャンネル範囲内のhippocampusの情報を皮質に流す。
        cerebral_cortex['now'] = self.consciousness
        cerebral_cortex['frames'] = self.frames
        
        # 認識・計画
        with mp.Pool(mp.cpu_count()-2) as pool:
            results = pool.starmap(self.think, [(cerebral_cortex, actor, emulator) for actor, emulator in zip(self.actors, self.emulators)])
        for result in results:
            actor_id, emulator_id, actions, images, rewards = result
            self._setupParams(actor_id=actor_id, emulator_id=emulator_id, actions=actions, images=images, rewards=rewards)

        # 行動の決定
        bestActor=max(self.predicted_reward, key=lambda k: self.predicted_reward.get(k))

        return bestActor, self.pairs[bestActor]
    
    def activator(self, _actObj):
        # if isinstance(_actObj, list): _actObj = self.actor(_actObj)
        return [act for act in _actObj.action if act in self.env_action_space]

    def episode(self):
        """
        Attribute:

        Returns:
            actors_reward: 各actorエージェントの獲得報酬
            emulators_reward: 各emulatorエージェントの獲得報酬
        """
        frame=0
        executor = ThreadPoolExecutor(max_workers=2)
        # state = _env.reset() # get initial state and prep environment
        thinker = executor.submit(self.thinker)
        # self.best = str(_elite_actor.team.id)
        total_reward = 0.
        actual_action = []
        flag = True
        pre_frame=-1
        while flag:
            if thinker.done(): # 意識的行動
                # thinking_actor.append(self.best)
                bestActor, pairEmulator = thinker.result()
                print(datetime.now(),frame, flag)
                assert bestActor in list(self.actions.keys()), f'dont has {bestActor} in {self.actions.keys()}'
                assert pairEmulator in list(self.memories.keys()), f'dont has {pairEmulator} in {self.memories.keys()}'
                if self.actions[bestActor] is []: flag=False
                

                for action, image in zip(self.actions[bestActor].action, self.memories[pairEmulator]):
                    frame+=1
                    state, reward, isDone, debug = self.env.step(action) # state : np.ndarray.flatten
                    if isDone: 
                        frame=self.frames
                        reward=0
                        self.state = self.env.reset()
                        flag=False

                    if not self.actor_scores.get(bestActor): self.actor_scores[bestActor]=0.
                    self.actor_scores[bestActor]+=reward

                    if not self.emulator_scores.get(pairEmulator): self.emulator_scores[pairEmulator]=0.
                    diff, unex = image.memory.compare(state.flatten(), reward)
                    self.emulator_scores[pairEmulator] += tanh(diff)
                    self.actual_states += [state.flatten()]
                    actual_action += [self.actor.actions]
                    self.actual_rewards += [reward]
                    self.unexpectancy  += [unex]
                    total_reward += reward
                    self.state = state

                thinker = executor.submit(self.thinker)
                if pre_frame==frame: 
                    print(f'frame is same',self.actions[bestActor], self.memories[pairEmulator])
                    flag=False
                    actual_action+=[np.nan]
                    break
                pre_frame=frame
                
            else: # 無意識的行動
                """
                Automata.hippocampusに明記
                """
                # actionCode = _elite_actor.act(EmulatorTPG.state.flatten())
                # if not ActionObject3.actions[actionCode] in range(_env.action_space.n): continue

                # EmulatorTPG.state, reward, isDone, debug = _env.step(ActionObject3.actions[actionCode])
                # total_reward += reward

                # if isDone: 
                #     EmulatorTPG.state = _env.reset()
                #     break
                # if _show:  self.show_state(_env, frame)
                pass

        thinker.cancel()
        return total_reward, actual_action

    def generation(self):

        _task   = self.env.spec.id
        self.setAgents()
        # breakpoint(type(emulators))
        print('start episode...')
        total_rewards=[]
        actual_actions=[]
        for _ in tqdm(range(self.episodes)):
            total_reward, actual_action = self.episode()
            total_rewards+=[total_reward]
            actual_actions+=[actual_action]
        print('... end episode')
        self.__class__.logger.info(f'rewards:{self.actor_scores}', extra={'className': self.__class__.__name__})

        # 報酬の贈与
        for actor in self.actors:
            actor.reward(score=self.actor_scores[actor.id], task=_task)

        # ここらへんのエミュレータの報酬設計
        for emulator in self.emulators:
            emulator.reward(score=self.emulator_scores[emulator.id], task=_task)

        self.actor.evolve([_task], _actionSequence=self.actual_actions, _actionReward=total_rewards)
        self.emulator.evolve([_task], _states=self.actual_states, _rewards=self.actual_rewards, _unexpectancies=self.unexpectancy)

        self.initParams()
        return total_rewards
    
    def story(self, _actor=None, _emulator=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=False, _test=False, _load=True, _dir=''):
        if _actor: 
            self.instance_valid(_actor)
            self.actor.trainer = _actor
        if _emulator: 
            self.instance_valid(_emulator)
            self.emulator.trainer = _emulator
        if _task:
            self.env = gym.make(_task)
        if _generations:
            self.generations=_generations
        if _episodes:
            self.episodes = _episodes
        if _frames:
            self.frames = _frames
        
        task = _dir+self.env.spec.id

        self.__class__.logger, filename = setup_logger(__name__, task, test=_test, load=_load)

        action_space = self.env.action_space
        observation_space = self.env.observation_space

        
        action = 2
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n

        state = observation_space.sample()
        self.setAction(action=action)
        self.setMemory(state=state.flatten())
        self.state = self.env.reset()
        def outHandler(signum, frame):
            if not _test: 
                self.actor.trainer.save(f'{task}/{filename}-act')
                self.emulator.trainer.save(f'{task}/{filename}-emu')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        for gen in tqdm(range(self.generations)): # generation loop
            # breakpoint(type(_emulator))
            scores = self.generation()
            # score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            self.__class__.logger.info(f'generation:{gen}, min:{min(scores)}, max:{max(scores)}, ave:{sum(scores)/len(scores)}', extra={'className': self.__class__.__name__})

        map(self.__class__.logger.removeHandler, self.__class__.logger.handlers)
        map(self.__class__.logger.removeFilter, self.__class__.logger.filters)

        return f'{task}/{filename}'

class Automata2(Automata1):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            cls._instance = True
            cls.Actor = Actor
            cls.Emulator = Emulator

            cls.Actor._logger=cls._logger
            cls.Emulator._logger=cls._logger
            # cls.ActionObject = _ActionObject
            # cls.MemoryObject = _MemoryObject
        return super().__new__(cls)
    
    def activator(self, _actObj):
        return [act for act in _actObj.action if act in self.env_action_space]

    def episode(self):
        """
        Attribute:

        Returns:
            actors_reward: 各actorエージェントの獲得報酬
            emulators_reward: 各emulatorエージェントの獲得報酬
        """
        frame=0
        executor = ThreadPoolExecutor(max_workers=2)
        # state = _env.reset() # get initial state and prep environment
        thinker = executor.submit(self.thinker)
        # self.best = str(_elite_actor.team.id)
        total_reward = 0.
        # thinking_actor = []
        while frame<self.frames:
            if thinker.done(): # 意識的行動
                # thinking_actor.append(self.best)
                bestActor, pairEmulator = thinker.result()
                assert bestActor in list(self.actions.keys()), f'dont has {bestActor} in {self.actions.keys()}'
                assert pairEmulator in list(self.memories.keys()), f'dont has {pairEmulator} in {self.memories.keys()}'
                

                for action, image in zip(self.actions[bestActor], self.memories[pairEmulator]):
                    frame+=1
                    if not action in range(self.env.action_space.n): continue
                    state, reward, isDone, debug = self.env.step(action) # state : np.ndarray.flatten
                    if isDone: 
                        frame=self.frames
                        reward=0

                    # assert self.actor_scores.get(bestActor), f'{bestActor} not in the {self.actor_scores}'
                    if not self.actor_scores.get(bestActor): self.actor_scores[bestActor]=0.
                    self.actor_scores[bestActor]+=reward

                    # if not self.actor_scores.get(bestActor): self.actor_scores[bestActor]=0.

                    # assert self.emulator_scores.get(pairEmulator), f'{pairEmulator} not in the {self.emulator_scores}'
                    if not self.emulator_scores.get(pairEmulator): self.emulator_scores[pairEmulator]=0.
                    diff, unex = image.memory.compare(state.flatten(), reward)
                    self.emulator_scores[pairEmulator] += tanh(diff)
                    self.actual_states += [state.flatten()]
                    self.actual_actions += [self.actor.actions]
                    self.actual_rewards += [reward]
                    self.unexpectancy  += [unex]
                    total_reward += reward
 
                    if isDone:
                        self.state=self.env.reset()
                        break

                    if self.show:  self.show_state(self.env, frame)
                    self.state = state
                    # print(frame)

                thinker = executor.submit(self.thinker)
            else: # 無意識的行動
                """
                Automata.hippocampusに明記
                """
                # actionCode = _elite_actor.act(EmulatorTPG.state.flatten())
                # if not ActionObject3.actions[actionCode] in range(_env.action_space.n): continue

                # EmulatorTPG.state, reward, isDone, debug = _env.step(ActionObject3.actions[actionCode])
                # total_reward += reward

                # if isDone: 
                #     EmulatorTPG.state = _env.reset()
                #     break
                # if _show:  self.show_state(_env, frame)
                pass

        thinker.cancel()
        return total_reward

    def generation(self):

        _task   = self.env.spec.id
        self.setAgents()
        # breakpoint(type(emulators))
        print('start episode...')
        # for _ in range(self.episodes):
        total_reward = self.episode()
        print('... end episode')
        self.logger.info(self.actor_scores) 
        
        # 報酬の贈与
        for actor in self.actors:
            actor.reward(score=self.actor_scores[actor.id], task=_task)

        # ここらへんのエミュレータの報酬設計
        for emulator in self.emulators:
            emulator.reward(score=self.emulator_scores[emulator.id], task=_task)

        self.actor.evolve([_task], _actionSequence=[self.actual_actions], _actionReward=[total_reward])
        self.emulator.evolve([_task], _states=self.actual_states, _rewards=self.actual_rewards, _unexpectancies=self.unexpectancy)

        self.initParams()
        return total_reward
    
    def story(self, _actor=None, _emulator=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=False, _test=False, _load=True, _dir=''):
        if _actor: 
            self.instance_valid(_actor)
            self.actor.trainer = _actor
        if _emulator: 
            self.instance_valid(_emulator)
            self.emulator.trainer = _emulator
        if _task:
            self.env = gym.make(_task)
        if _generations:
            self.generations=_generations
        if _episodes:
            self.episodes = _episodes
        if _frames:
            self.frames = _frames
        
        task = _dir+self.env.spec.id

        self.setup_logger(__name__, task, test=_test, load=_load)

        action_space = self.env.action_space
        observation_space = self.env.observation_space

        
        action = 2
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n

        state = observation_space.sample()
        self.setAction(action=action)
        self.setMemory(state=state.flatten())
        self.state = self.env.reset()
        def outHandler(signum, frame):
            if not _test: 
                self.actor.trainer.save(f'{task}/{self.filename}-act')
                self.emulator.trainer.save(f'{task}/{self.filename}-emu')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        summaryScores = []

        tStart = time.time()
        for gen in tqdm(range(self.generations)): # generation loop
            # breakpoint(type(_emulator))
            total_score = self.generation()
            # score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            self.logger.info(f'generation:{gen}, score:{total_score}', extra={'className': self.__class__.__name__})
            summaryScores.append(total_score)

        map(self.logger.removeHandler, self.logger.handlers)
        map(self.logger.removeFilter, self.logger.filters)

        return f'{task}/{self.filename}'

class Automata3(Automata2):
    """ type 3
        conscious: operation to output channel
            activator: manage output for environment
            recollector: manage hippocampus operation
    
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            from _tpg.trainer import Trainer2_3, Trainer1_3
            cls._instance = True
            cls.Cerebellum = Trainer1_3
            cls.Cerebrum = Trainer1_3
            cls.Hippocampus = Trainer2_3
            
        return super().__new__(cls)

    def __init__(self):
        
        # setup trainers
        self.hippocampus = self.__class__.Hippocampus()
        self.cerebellum = self.__class__.Cerebellum(hippocampus=self.hippocampus)
        self.cerebrum = self.__class__.Cerebrum(hippocampus=self.hippocampus)
        self.tasks = set()
        self.env = None
        self.state = None

    def setEnv(self, _env):
        self.env = _env
        self.state = _env.reset()
        self.tasks.add(_env.spec.id)

    def activator(self, _actObj):
        return super().activator(_actObj)
    
    def recollector(self, _actObj):
        return self.hippocampus.operator(_actObj)

    def consciousness(self, willness):
        if willness[-1].ch == 0:
            self.activator(willness[:-1])
        else:
            self.recollector(willness[:-1])

    def sleep(self):
        self.hippocampus.evolve(list(self.tasks))
        self.cerebellum.evolve(list(self.tasks))
        self.cerebrum.evolve(list(self.tasks))

class Automata(_Automata):
    Hippocampus=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.memory_object import Hippocampus
            cls._instance = True
            cls.Actor = Actor
            cls.Emulator = Emulator
            cls.Hippocampus = Hippocampus

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        actions=None,
        states=None,
        teamPopSize:int=10,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pActMut:float=0.1,                  # *
        pActAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4,
        thinkingTime=7,
    ):
        # super().__init__()
        self.actor = self.__class__.Actor(
            actions=actions,
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pActMut=pActMut,                  # *
            pActAtom=pActAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nActRegisters=nActRegisters
        )
        self.emulator = self.__class__.Emulator(
            state=states,
            teamPopSize=teamPopSize,               # *
            rootBasedPop=rootBasedPop,             
            gap=gap,                      
            inputSize=inputSize,                
            nRegisters=nRegisters,                   # *
            initMaxTeamSize=initMaxTeamSize,             # *
            initMaxProgSize=initMaxProgSize,             # *
            maxTeamSize=maxTeamSize,                 # *
            pLrnDel=pLrnDel,                  # *
            pLrnAdd=pLrnAdd,                  # *
            pLrnMut=pLrnMut,                  # *
            pProgMut=pProgMut,                 # *
            pMemMut=pActMut,                  # *
            pMemAtom=pActAtom,                # *
            pInstDel=pInstDel,                 # *
            pInstAdd=pInstAdd,                 # *
            pInstSwp=pInstSwp,                 # *
            pInstMut=pInstMut,                 # *
            doElites=doElites, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nMemRegisters=nActRegisters
        )
        self.thinkingTimeLimit=thinkingTime

        self.generations=100
        self.episodes=1
        self.frames = 500
        self.show = False
        self.logger = None
        self.initParams()

    def _setupParams(self, actor_id, emulator_id, actions=None, images=None, rewards=None):
        self.actions[actor_id]=[]
        # self.predicted_reward[actor_id]=0.
        if not self.predicted_reward.get(actor_id): self.predicted_reward[actor_id]=0.
        if rewards is not None: self.predicted_reward[actor_id] = sum(rewards)
        if actions is not None: self.actions[actor_id] = actions

        self.pairs[actor_id] = emulator_id
        if not self.memories.get(emulator_id): self.memories[emulator_id]=[]
        if images is not None: self.memories[emulator_id] += images

    def setAction(self, action):
        self.actor.setActions(action)

    def setMemory(self, state):
        self.emulator.setMemories(state.flatten())

    def setAgents(self):
        """
        wake up
        reset hippocampus
        """
        self.hippocampus = self.Hippocampus()
        self.actors = self.actor.getAgents()
        self.elite_act = self.actor.getEliteAgent()
        self.emulators = self.emulator.getAgents()
        self.elite_emu = self.emulator.getEliteAgent()

    def setEnv(self, _env):
        self.env = _env
        self.task = _env.spec.id
        self.state = _env.reset()

    def instance_valid(self, _actor=None, _emulator=None) -> None:
        assert self.__class__.Actor.Trainer is not self.__class__.Emulator.Trainer

        if _actor: 
            assert isinstance(_actor, self.__class__.Actor.Trainer), f'this actor is not {self.__class__.Actor}'
        if _emulator:
            assert isinstance(_emulator, self.__class__.Emulator.Trainer), f'this emulator is not {self.__class__.Emulator}'

    def think(self, cerebral_cortex, _actor, _emulator):
        self.__class__.Emulator.Trainer.MemoryObject.memories = cerebral_cortex['memories']
        self.__class__.Actor.Trainer.ActionObject.actions = cerebral_cortex['actions']
        state = np.array(cerebral_cortex['image']).flatten()

        assert isinstance(_actor, self.__class__.Actor.Trainer.Agent)
        assert isinstance(_emulator, self.__class__.Emulator.Trainer.Agent)
        assert self.__class__.Actor.Trainer.ActionObject.actions is not None
        assert self.__class__.Emulator.Trainer.MemoryObject.memories is not None


        # timeout_start = time.time()
        # while (time.time() - timeout_start) < self.thinkingTimeLimit:
        assumption = []
        actObj = _actor.act(state) # actionObject
        reward = 0.
        for act in actObj.action:
            memObj = _emulator.image(act, state) # memoryObject
            assumption += [memObj]
            reward += memObj.reward
            state = memObj.recall(state)
        
        return actObj, assumption, reward, _actor.id, _emulator.id

    def thinker(self):
        """意識にとって何が最善の行動となるかの選別
        Returns:
            new_act:
            act_filter:
            cons_filter:
        """
        manager = mp.Manager()
        cerebral_cortex = manager.dict()

        # 脳皮質内の信号
        cerebral_cortex['actions']  = self.actions
        cerebral_cortex['memories'] = self.memories
        # determined actionによって制限された意識チャンネル範囲内のhippocampusの情報を皮質に流す。
        cerebral_cortex['image'] = self.focus & self.hippocampus() # new memoryObj
        
        # 認識・計画
        with mp.Pool(mp.cpu_count()-2) as pool:
            short_memory = pool.starmap(self.think, [(cerebral_cortex, actor, emulator) for actor, emulator in zip(self.actors, self.emulators)])
            # (actor_id, emulator_id, [...memoryCode]),

        # mind tuning reward differ with mind bias
        compare = []
        for plan in short_memory:
            compare += [plan[2]]

        best=max(range(len(compare)), key=compare.__getitem__)

        # 追認説的な思考プロセスの想起
        return short_memory[best] # (actObj, [...memObj], predicted_reward)

    def episode(self):
        """
        記憶単位
        行動と報酬のエピソードを生成。
        エピソードはAutomata.hippocampusに電気的記憶として保存される。
        睡眠期と行動期を分ける？
        """
        frame=0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.thinking = self.executor.submit(self.thinker)

        total_reward = 0.
        while frame<self.frames:
            actionSequence = self.activator(self.consciousness+self.unconsciousness) # return actionObj + actionObj = new actionObj
            memorySequence = []
            rewardSequence = []
            for action in actionSequence:
                # action = self.actor.actions[actioinCode]
                state, reward, isDone, debug = self.env.step(action)
                memorySequence+=[state.flatten()]
                rewardSequence+=[reward]
                if isDone:
                    frame=self.frames
                    break
                frame+=1
            self.hippocampus.real.append(actionSequence, memorySequence, rewardSequence)
            self.state = state
            
        self.thinking.cancel()

        return total_reward

    def sleep(self, _task=None):
        """
        TODO: treat hippocampus reward?
        TODO: treat actor and emulator evolve
        HOW: append memoryobject action, image
        """
        # 報酬の贈与
        for actor in self.actors:
            actor.reward(score=self.actor_scores[actor.id], task=_task if _task else self.task)

        # エミュレータの報酬設計
        for emulator in self.emulators:
            emulator.reward(score=self.emulator_scores[emulator.id], task=_task if _task else self.task)

        self.actor.evolve([_task])
        self.emulator.evolve([_task])

    def generation(self):
        self.setAgents()
        self.episode()
        self.sleep()
        self.initParams()
    
    def story(self, _actor=None, _emulator=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=False, _test=False, _load=True, _dir=''):
        if _actor: 
            self.instance_valid(_actor)
            self.actor.trainer = _actor
        if _emulator: 
            self.instance_valid(_emulator)
            self.emulator.trainer = _emulator
        if _task:
            self.env = gym.make(_task)
        if _generations:
            self.generations=_generations
        if _episodes:
            self.episodes = _episodes
        if _frames:
            self.frames = _frames
        
        task = _dir+self.env.spec.id

        logger, filename = setup_logger(__name__, task, test=_test, load=_load)

        action_space = self.env.action_space
        observation_space = self.env.observation_space

        
        action = 2
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n

        state = observation_space.sample()
        self.setAction(action=action)
        self.setMemory(state=state.flatten())

        def outHandler(signum, frame):
            if not _test: 
                self.actor.trainer.save(f'{task}/{filename}-act')
                self.emulator.trainer.save(f'{task}/{filename}-emu')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        for gen in tqdm(range(self.generations)): # generation loop
            # breakpoint(type(_emulator))
            self.generation()
            # score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))
            total_score = self.hippocampus.real.score

            logger.info(f'generation:{gen}, score:{total_score}', extra={'className': self.__class__.__name__})

        map(logger.removeHandler, logger.handlers)
        map(logger.removeFilter, logger.filters)

        return filename
    
    def activator(self, _actObj):
        return [act for act in _actObj.action if act in range(self.env.action_space.n)]

    @property
    def consciousness(self): # return self.actor.ActionObject
        if self.thinking.done():
            actObj, memObjs, emotion, bestActor, pairEmulator = self.thinking.result()
            self.hippocampus.mind.append(actObj, memObjs, emotion)
            self.thinking = self.executor.submit(self.thinker)
            if actObj.action[0] is not np.nan:
                return self.intention & actObj # filterd action object self.intention is NaN*int
            self.focus = actObj.action.fragment 

        return self.actor.NaN

    @property
    def unconsciousness(self): # # return self.actor.ActionObject
        return self.elite.act(self.state.flatten())
 