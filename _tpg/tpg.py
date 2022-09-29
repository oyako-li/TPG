from math import tanh
from _tpg.base_log import setup_logger
from _tpg.utils import breakpoint
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import multiprocessing as mp
import tkinter as tk
import numpy as np
import sys
import gym
import signal
import time

class _TPG:
    Trainer=None
    _instance=None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            from _tpg.trainer import _Trainer

            cls._instance = True
            cls.Trainer = _Trainer

        return super().__new__(cls)

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
            # memType=memType, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            # operationSet=operationSet, 
            # traversal=traversal, 
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nActRegisters=nActRegisters)
        
        self.generations=10
        self.episodes=1
        self.frames = 500
        self.show = False
        self.logger = None

    def setActions(self, actions):
        self.actions = self.trainer.setActions(actions)

    def setEnv(self, env):
        self.env = env

    def getAgents(self):
        return self.trainer.getAgents()
    
    def setAgents(self, task='task'):
        self.agents = self.trainer.getAgents(task=task)

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

    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
        
        for agent in self.agents: # to multi-proccess
            state = self.env.reset() # get initial state and prep environment
            score = 0
            for i in range(self.frames): # run episodes that last 500 frames
                act = agent.act(state)
                if not act in range(self.env.action_space.n): continue
                state, reward, isDone, debug = self.env.step(act)
                score += reward # accumulate reward in score

                if isDone: break # end early if losing state
                if self.show:self.flush_render(i)

            if _scores.get(agent.id) is None : _scores[agent.id]=0
            _scores[agent.id] += score # store score

            # if self.logger is not None: self.logger.info(f'{_id},{score}')

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores

    def generation(self):
        _scores = {}
        _task = self.env.spec.id
        self.agents = self.trainer.getAgents()
        for _ in range(self.episodes):     
            _scores = self.episode(_scores)
        for i in _scores:               
            _scores[i]/=self.episodes
        for agent in self.agents: 
            agent.reward(_scores[str(agent.team.id)],task=_task)
        self.trainer.evolve([_task])

        return _scores
    
    def growing(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        if _trainer: 
            self.instance_valid(_trainer)
            self.trainer = _trainer
        
        if _task:
            self.env = gym.make(_task)
        
        
        task = _dir+self.env.spec.id

        logger, filename = setup_logger(__name__, task, test=_test, load=_load)
        self.logger = logger
        self.generations = _generations
        self.episodes = _episodes
        self.frames = _frames
        self.show = _show


        action_space = self.env.action_space
        action = 0
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n
        self.trainer.setActions(actions=action)

        def outHandler(signum, frame):
            if not _test: self.trainer.save(f'{task}/{filename}')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        if self.show: 
            self.set_tk_render()
        

        for gen in range(_generations): # generation loop
            scores = self.generation()

            self.logger.info(f'generation:{gen}, min:{min(scores.values())}, max:{max(scores.values())}, ave:{sum(scores.values())/len(scores)}')

        list(map(self.logger.removeHandler, self.logger.handlers))
        list(map(self.logger.removeFilter, self.logger.filters))

        return f'{task}/{filename}'

    def start(self, _task, _show, _test, _load, _trainer=None, _generations=1000, _episodes=1, _frames=500):
        if _trainer is None : 
            if not self.trainer : raise Exception('trainer is not defined')
            _trainer = self.trainer

        _filename = self.growing(_trainer, _task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load)
        if not _test: _trainer.saveToFile(f'{_task}/{_filename}')
        return _filename
    
    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, self.__class__.Trainer): raise Exception(f'this object is not {self.__class__.Trainer}')

class MHTPG(_TPG):
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            from _tpg.trainer import Trainer1
            cls._instance = True
            cls.Trainer = Trainer1

        return super().__new__(cls, *args, **kwargs)

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self.trainer.evolve(tasks, multiTaskType, extraTeams)

class ActorTPG(MHTPG):
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            from _tpg.trainer import Trainer4
            cls._instance = True
            cls.Trainer = Trainer4

        return super().__new__(cls, *args, **kwargs)

    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
        
        for agent in self.agents: # to multi-proccess
            state = self.env.reset() # get initial state and prep environment
            score = 0
            frame = 0
            
            while frame<self.frames: # run episodes that last 500 frames
                actionCode = agent.act(state)
                for action in self.actions[actionCode]:
                    frame+=1
                    if not action in range(self.env.action_space.n):continue
                    state, reward, isDone, debug = self.env.step(action)
                    score += reward # accumulate reward in score

                    if isDone: break # end early if losing state
                    if self.show:self.flush_render()

            if _scores.get(agent.id) is None : _scores[agent.id]=0
            _scores[agent.id] += score # store score

        return _scores

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
        self.logger = None

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _states=None, _unexpectancies=None):
        self.trainer.evolve(tasks, multiTaskType, extraTeams, _states, _unexpectancies)

    def setMemories(self, states):
        self.memories = self.trainer.setMemories(states.flatten())
    
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
                imageCode = agent.image(act, state.flatten())
                state, reward, isDone, debug = self.env.step(act)
                diff, unex = self.memories[imageCode].memorize(state.flatten(), reward)

                score += tanh(np.power(diff, 2).sum())
                states+=[state.flatten()]
                unexpectancies+=[unex]

                if isDone: break # end early if losing state
                if self.show: self.show_state(self.env, _)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

            # if self.logger is not None: self.logger.info(f'{_id},{score}')

        return _scores, states, unexpectancies

    def generation(self):
        _scores = {}
        self.agents = self.trainer.getAgents()
        _task = self.env.spec.id
        for _ in range(self.episodes):  _scores, states, unexpectancies = self.episode(_scores)
        for i in _scores:               _scores[i]/=self.episodes
        for agent in self.agents:       agent.reward(_scores[agent.id], task=_task)
        self.trainer.evolve([_task], _states=states, _unexpectancies=unexpectancies)

        return _scores
   
    def growing(self, _trainer=None, _task:str=None, _generations:int=100, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        if _trainer: 
            self.instance_valid(_trainer)
            self.trainer = _trainer
        
        if _task:
            self.env = gym.make(_task)
        
        task = _dir+self.env.spec.id+'-em'

        logger, filename = setup_logger(__name__, task, test=_test, load=_load)
        self.logger = logger
        self.generations = _generations
        self.episodes = _episodes
        self.frames = _frames
        self.show = _show

        self.trainer.setMemories(state=self.env.observation_space.sample().flatten())

        def outHandler(signum, frame):
            if not _test: self.trainer.save(f'{task}/{filename}')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        for gen in tqdm(range(_generations)): # generation loop
            scores = self.generation()
            self.logger.info(f'generation:{gen}, min:{min(scores.values())}, max:{max(scores.values())}, ave:{sum(scores.values())/len(scores)}')

        list(map(self.logger.removeHandler, self.logger.handlers))
        list(map(self.logger.removeFilter, self.logger.filters))

        return f'{task}/{filename}'

class EmulatorTPG1(EmulatorTPG):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.trainer import Trainer3
            cls._instance = True
            cls.Trainer = Trainer3

        return super().__new__(cls, *args, **kwargs)

class Automata(_TPG):
    Actor=None
    Emulator=None
    hippocampus=None
    _instance=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            from _tpg.memory_object import _Memory
            cls._instance = True
            cls.Actor = MHTPG
            cls.Emulator = EmulatorTPG
            cls.hippocampus = _Memory()
            # cls.ActionObject = _ActionObject
            # cls.MemoryObject = _MemoryObject
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

        self.generations=100
        self.episodes=1
        self.frames = 500
        self.show = False
        self.logger = None
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

    def setAction(self, action):
        self.actor.setActions(action)

    def setMemory(self, state):
        self.emulator.setMemories(state.flatten())

    def setAgents(self):
        """set hippocampus"""
        self.actor_scores = {}
        self.emulator_scores = {}
        self.actual_states = []
        self.unexpectancy = []
        self.actors = self.actor.getAgents()
        self.emulators = self.emulator.getAgents()
        for actor in self.actors:
            self.actor_scores[actor.id] = 0.
            assert not self.actor_scores.get(actor.id), f'{actor.id} cant assaign {self.actor_scores}'
        for emulator in self.emulators:
            self.emulator_scores[emulator.id] = 0.
            assert not self.emulator_scores.get(emulator.id)

    def setEnv(self, _env):
        self.env = _env

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

    @property
    def consciousness(self):
        # self.consciousness_channel_key = []
        # return self.__class__.hippocampus(self.consciousness_channel_key)
        return self.state.flatten()

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
        while (time.time() - timeout_start) < self.thinkingTimeLimit:
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
        cerebral_cortex['actions']  = self.__class__.Actor.Trainer.ActionObject.actions
        cerebral_cortex['memories'] = self.__class__.Emulator.Trainer.MemoryObject.memories
        # determined actionによって制限された意識チャンネル範囲内のhippocampusの情報を皮質に流す。
        cerebral_cortex['now'] = self.consciousness
        
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
                    self.unexpectancy  += [unex]
                    total_reward += reward
 
                    if isDone:
                        self.state=self.env.reset()
                        break

                    if self.show:  self.show_state(self.env, frame)
                    self.state = state

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
        for _ in range(self.episodes):
            total_reward = self.episode()

        # 報酬の贈与
        for actor in self.actors:
            actor.reward(score=self.actor_scores[actor.id], task=_task)

        # ここらへんのエミュレータの報酬設計
        for emulator in self.emulators:
            emulator.reward(score=self.emulator_scores[emulator.id], task=_task)

        self.actor.evolve([_task])
        self.emulator.evolve([_task], _states=self.actual_states, _unexpectancies=self.unexpectancy)

        self.initParams()
        return total_reward
    
    def growing(self, _actor=None, _emulator=None, _task:str=None, _generations:int=None, _episodes:int=None, _frames:int=None, _show=False, _test=False, _load=True, _dir=''):
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

            logger.info(f'generation:{gen}, score:{total_score}')
            summaryScores.append(total_score)

        #clear_output(wait=True)
        # logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
        # logger.info(f'Results: Min, Max, Avg, {summaryScores}')
        map(logger.removeHandler, logger.handlers)
        map(logger.removeFilter, logger.filters)

        return filename
