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
    
    def setAgents(self):
        self.agents = self.trainer.getAgents()

    
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
            _id = str(agent.team.id)
            for i in range(self.frames): # run episodes that last 500 frames
                act = agent.act(state)
                if not act in range(self.env.action_space.n): continue
                state, reward, isDone, debug = self.env.step(act)
                score += reward # accumulate reward in score

                if isDone: break # end early if losing state
                if self.show:self.flush_render(i)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

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

    def setMemories(self, states):
        self.memories = self.trainer.setMemories(states)
    
    def episode(self, _scores):
        assert self.trainer is not None and self.env is not None, 'You should set Actioins'
      
        states = []
        unexpectancies = []
        for agent in self.agents: # to multi-proccess
            
            state = self.env.reset() # get initial state and prep environment
            score = 0
            _id = str(agent.team.id)
            for _ in range(self.frames): # run episodes that last 500 frames
                act = self.env.action_space.sample()
                imageCode = agent.image(act, state.flatten())
                state, reward, isDone, debug = self.env.step(act)
                diff, unex = self.__class__.Trainer.MemoryObject.memories[imageCode].memorize(state.flatten(), reward)

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
        for agent in self.agents:       agent.reward(_scores[str(agent.team.id)], task=_task)
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

class Automata(_TPG):
    Actor=None
    Emulator=None
    # ActionObject=None
    # MemoryObject=None
    _instance=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # from _tpg.action_object import _ActionObject
            # from _tpg.memory_object import _MemoryObject
            cls._instance = True
            cls.Actor = MHTPG
            cls.Emulator = EmulatorTPG
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
        thinkingTime=30,
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
        self.logger = None
        self.initParams()

    def _setupParams(self, actor_id, emulator_id, actions=None, images=None, rewards=None):
        self.actions[actor_id]=[]
        # self.rewards[actor_id]=0.
        if not self.rewards.get(actor_id) : self.rewards[actor_id]=0.
        if rewards is not None: self.rewards[actor_id] = sum(rewards)
        if actions is not None: self.actions[actor_id] = actions

        self.pairs[actor_id] = emulator_id
        if not self.emulator.memories.get(emulator_id) : self.emulator.memories[emulator_id]=[]
        if images is not None: self.emulator.memories[emulator_id]   += images

    def setAction(self, action):
        self.actor.setActions(action)

    def setMemory(self, state):
        self.emulator.setMemories(state)

    def setAgents(self):
        self.actors = self.actor.getAgents()
        self.emulators = self.emulator.getAgents()

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
        self.rewards    = {}

    def think(self, hippocampus, _actor, _emulator):
        # self.__class__.Emulator.Trainer.MemoryObject.memories = hippocampus['memories']
        # self.__class__.Actor.Trainer.ActionObject.actions = hippocampus['actions']
        assert isinstance(_actor, self.__class__.Actor.Trainer.Agent)
        assert isinstance(_emulator, self.__class__.Emulator.Trainer.Agent)
        assert self.__class__.Actor.Trainer.ActionObject.actions is not None
        assert self.__class__.Emulator.Trainer.MemoryObject.memories is not None

        state = np.array(hippocampus['now'])
        actor_id = str(_actor.team.id)
        emulator_id = str(_emulator.team.id)
        actionCodes = []
        memoryCodes = []
        rewards     = []
        # timeout_start = time.time()
        for i in range(hippocampus['frame']):
            actionCode = _actor.act(state)
            imageCode  = _emulator.image(actionCode, state)
            actionCodes  += [actionCode]
            memoryCodes  += [imageCode]
            rewards += [hippocampus['memories'][imageCode].reward]
            state = hippocampus['memories'][imageCode].recall(state)
            # breakpoint(self.thinkingTimeLimit)
        return actor_id, emulator_id, actionCodes, memoryCodes, rewards

    def thinker(self):
        manager = mp.Manager()
        hippocampus = manager.dict()
        hippocampus['actions']  = self.__class__.Actor.Trainer.ActionObject.actions
        hippocampus['memories'] = self.__class__.Emulator.Trainer.MemoryObject.memories
        hippocampus['now'] = self.state
        hippocampus['frame']= self.frames
        
        with mp.Pool(mp.cpu_count()-2) as pool:
            # params = [(actor, emulator) for actor, emulator in zip(_actors, _emulators)]
            results = pool.starmap(self.think, [(hippocampus, actor, emulator) for actor, emulator in zip(self.actors, self.emulators)])
        for result in results:
            actor_id, emulator_id, actions, images, rewards = result
            self._setupParams(actor_id=actor_id, emulator_id=emulator_id, actions=actions, images=images, rewards=rewards)
            assert self.pairs.get(actor_id), (self.pairs[actor_id], emulator_id)
        bestActor=max(self.rewards, key=lambda k: self.rewards.get(k))

        return bestActor, self.pairs[bestActor]

    def episode(self, _scores:dict, _states:dict):
        # _scores = {}
        # _states = {}
        frame=0
        executor = ThreadPoolExecutor(max_workers=2)
        # state = _env.reset() # get initial state and prep environment
        thinker = executor.submit(self.thinker)
        # self.best = str(_elite_actor.team.id)
        total_reward = 0.
        # thinking_actor = []
        while frame<self.frames:
            if thinker.done():
                scores = []
                states = []
                # thinking_actor.append(self.best)
                bestActor, pairEmulator = thinker.result()
                for actionCode in self.actions[bestActor]:
                    frame+=1
                    if not self.actor.actions[actionCode] in range(self.env.action_space.n): continue
                    state, reward, isDone, debug = self.env.step(self.actor.actions[actionCode]) # state : np.ndarray.flatten
                    scores+=[reward] # accumulate reward in score
                    total_reward += reward
                    # print(reward)
                    states.append(state.flatten())
                    if isDone: 
                        frame=self.frames
                        self.state=self.env.reset()
                        break
                        # end early if losing state
                    if self.show:  self.show_state(self.env, frame)
                    self.state = state
                
                # breakpoint('thinker ok')
                if _scores.get(bestActor)    is None : _scores[bestActor]=[]
                if _states.get(pairEmulator) is None : 
                    assert pairEmulator, pairEmulator
                    _states[pairEmulator]=[]
                _scores[bestActor]    += scores    # store score
                _states[pairEmulator] += states    # store states
                thinker = executor.submit(self.thinker)
            else:
                # actionCode = _elite_actor.act(EmulatorTPG.state.flatten())
                # if not ActionObject3.actions[actionCode] in range(_env.action_space.n): continue

                # EmulatorTPG.state, reward, isDone, debug = _env.step(ActionObject3.actions[actionCode])
                # total_reward += reward

                # if isDone: 
                #     EmulatorTPG.state = _env.reset()
                #     break
                # if _show:  self.show_state(_env, frame)
                pass

            # breakpoint('thinking is done')
        # for activate in thinking_actor:

        # if _logger is not None: _logger.info(f'this time score:{total_reward}')
        thinker.cancel()

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores, _states, total_reward

    def generation(self):

        _scores = {}
        _states = {}
        _task   = self.env.spec.id
        self.setAgents()
        # breakpoint(type(emulators))
        for _ in range(self.episodes):
            _scores, _states, total_reward = self.episode(_scores, _states)

        reward_for_actor = {}
        states=[]

        unexpectancy=0.
        unexpectancies=[]
        re_total=0.
        for key, val in _scores.items():
            re_total += sum(val)

        for ac in _scores:               
            # total = sum(re_total)#-self.rewards[ac]
            # total = tanh(total)
            # total-= tanh(self.rewards[ac])
            reward_for_actor[ac] = re_total
            em = self.pairs[ac]
            score = 0.
            assert len(_states[em])==len(_scores[ac])
            # unexpectancyが予想よりいいか悪いかを表す。
            # unexpectancy = abs(total)

            for state, imageCode, reward in zip(_states[em], self.memories[em], _scores[ac]):
                diff, unex = self.emulator.memories[imageCode].memorize(state, reward)
                # print(reward)
                score += tanh(np.power(diff, 2).sum())
                states+=[state]
                unexpectancies+=[unex]
            _states[em]=score
            # 予想外度　-1~1ぐらい、予想外度が小さいとあまりスコアを得られない。
            # ただ、エミュレータの場合、予想外度が小さいものほど評価されるべき。
            # 報酬予想が正しいものが残る。
            # 報酬予測が予想外のものほど、短期的に、銘記されやすい。
            # pop up されやすい。
            # 逆に、予想外のチームはスコアは得られにくい。

                
        # 報酬の贈与
        for actor in self.actors:
            if reward_for_actor.get(str(actor.team.id)) : 
                actor.reward(reward_for_actor[str(actor.team.id)], task=_task)
        # ここらへんのエミュレータの報酬設計
        for emulator in self.emulators:
            if _states.get(str(emulator.team.id)): 
                emulator.reward(_states[str(emulator.team.id)], task=_task)
        # breakpoint()
        self.actor.trainer.evolve([_task])
        # breakpoint('finish')
        self.emulator.trainer.evolve([_task], _states=states, _unexpectancies=unexpectancies)
        # breakpoint('after evolving')
        # print(unexpectancies)

        self.initParams()

        return total_reward
    
    def growing(self, _actor=None, _emulator=None, _task:str=None, _generations:int=1000, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True, _dir=''):
        if _actor: 
            self.instance_valid(_actor)
            self.actor.trainer = _actor
        if _emulator: 
            self.instance_valid(_emulator)
            self.emulator.trainer = _emulator
        if _task:
            self.env = gym.make(_task)
        
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
        for gen in tqdm(range(_generations)): # generation loop
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
