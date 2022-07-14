import asyncio
from math import tanh
from _tpg.action_object import ActionObject1, ActionObject3
from _tpg.agent import Agent1, Agent3, Agent2
from _tpg.trainer import Trainer, Trainer1, Trainer3, Trainer2, loadTrainer
from _tpg.memory_object import MemoryObject
from _tpg.base_log import setup_logger
from _tpg.utils import breakpoint
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import queue
import numpy as np
import sys
import gym
import signal
import time

class TPG:

    def __init__(self, 
        actions=2,
        teamPopSize:int=1000,               # *
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
        memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4,
    ): pass

    def show_state(self,env, step=0, name='', info=''):
        plt.figure(3)
        plt.clf()
        plt.imshow(env.render(mode='rgb_array'))
        plt.title("%s | Step: %d %s" % (name, step, info))
        plt.axis('off')

    def getState(self,inState):
        # each row is all 1 color
        rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

        # add each with appropriate shifting
        # get RRRRRRRR GGGGGGGG BBBBBBBB
        return np.add(np.left_shift(rgbRows[0], 16),
            np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))

    def episode(self,_agents, _env, _logger=None, _scores={}, _frames:int=500, _show=False):
        
        for agent in _agents: # to multi-proccess
            
            state = _env.reset() # get initial state and prep environment
            score = 0
            _id = str(agent.team.id)
            for _ in range(_frames): # run episodes that last 500 frames
                act = agent.act(state)
                # feedback from env
                if not act in range(_env.action_space.n): continue
                state, reward, isDone, debug = _env.step(act)
                score += reward # accumulate reward in score

                if isDone: break # end early if losing state
                if _show:
                    self.show_state(_env, _)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

            if _logger is not None: _logger.info(f'{_id},{score}')

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores

    def generation(self,_trainer, _env, _logger=None, _episodes=1, _frames= 500, _show=False):
        _scores = {}
        agents = _trainer.getAgents()
        _task = _env.spec.id
        for _ in range(_episodes):      _scores = self.episode(agents, _env, _logger=_logger, _scores=_scores, _frames=_frames, _show=_show)
        for i in _scores:                _scores[i]/=_episodes
        for agent in agents:            agent.reward(_scores[str(agent.team.id)],task=_task)
        _trainer.evolve([_task])

        return _scores
    
    def growing(self, _trainer, _task:str, _generations:int=1000, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True):
        self.instance_valid(_trainer)
        logger, filename = setup_logger(__name__, _task, test=_test, load=_load)
        # print(_task,filename)
        env = gym.make(_task) # make the environment
        action_space = env.action_space
        action = 0
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n
        _trainer.resetActions(actions=action)

        def outHandler(signum, frame):
            if not _test: _trainer.saveToFile(f'{_task}/{filename}')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        

        summaryScores = []

        tStart = time.time()
        for gen in tqdm(range(_generations)): # generation loop
            scores = self.generation(_trainer, env, logger, _episodes=_episodes, _frames=_frames, _show=_show)

            score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            logger.info(f'generation:{gen}, score:{score}')
            summaryScores.append(score)
            
        #clear_output(wait=True)
        logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
        logger.info(f'Results: Min, Max, Avg, {summaryScores}')
        return filename

    def instance_valid(self, trainer)->bool: pass

    def start(self, _task, _show, _test, _load, _trainer=None, _generations=1000, _episodes=1, _frames=500):
        if _trainer is None : 
            if not self.trainer : raise Exception('trainer is not defined')
            _trainer = self.trainer

        _filename = self.growing(_trainer, _task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load)
        if not _test: _trainer.saveToFile(f'{_task}/{_filename}')
        return _filename

class NativeTPG(TPG):

    def __init__(self, actions=2, teamPopSize: int = 1000, rootBasedPop: bool = True, gap: float = 0.5, inputSize: int = 33600, nRegisters: int = 8, initMaxTeamSize: int = 10, initMaxProgSize: int = 10, maxTeamSize: int = -1, pLrnDel: float = 0.7, pLrnAdd: float = 0.6, pLrnMut: float = 0.2, pProgMut: float = 0.1, pActMut: float = 0.1, pActAtom: float = 0.95, pInstDel: float = 0.5, pInstAdd: float = 0.4, pInstSwp: float = 0.2, pInstMut: float = 1.0, doElites: bool = True, memType="def", memMatrixShape: tuple = (100, 8), rampancy: tuple = (0, 0, 0), operationSet: str = "custom", traversal: str = "team", prevPops=None, mutatePrevs=True, initMaxActProgSize: int = 6, nActRegisters: int = 4):
        self.trainer = Trainer(
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
            memType=memType, 
            memMatrixShape=memMatrixShape,       # *
            rampancy=rampancy,
            operationSet=operationSet, 
            traversal=traversal, 
            prevPops=prevPops, mutatePrevs=mutatePrevs,
            initMaxActProgSize=initMaxActProgSize,           # *
            nActRegisters=nActRegisters)

    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, Trainer): raise Exception('this object is not Trainer')

class MemoryAndHierarchicalTPG(TPG):

    def __init__(self, actions=2, teamPopSize: int = 1000, rootBasedPop: bool = True, gap: float = 0.5, inputSize: int = 33600, nRegisters: int = 8, initMaxTeamSize: int = 10, initMaxProgSize: int = 10, maxTeamSize: int = -1, pLrnDel: float = 0.7, pLrnAdd: float = 0.6, pLrnMut: float = 0.2, pProgMut: float = 0.1, pActMut: float = 0.1, pActAtom: float = 0.95, pInstDel: float = 0.5, pInstAdd: float = 0.4, pInstSwp: float = 0.2, pInstMut: float = 1.0, doElites: bool = True, memType="def", memMatrixShape: tuple = (100, 8), rampancy: tuple = (0, 0, 0), operationSet: str = "custom", traversal: str = "team", prevPops=None, mutatePrevs=True, initMaxActProgSize: int = 6, nActRegisters: int = 4):
        self.trainer = Trainer1(actions, teamPopSize, rootBasedPop, gap, inputSize, nRegisters, initMaxTeamSize, initMaxProgSize, maxTeamSize, pLrnDel, pLrnAdd, pLrnMut, pProgMut, pActMut, pActAtom, pInstDel, pInstAdd, pInstSwp, pInstMut, doElites, memType, memMatrixShape, rampancy, operationSet, traversal, prevPops, mutatePrevs, initMaxActProgSize, nActRegisters)

    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, Trainer1): raise Exception('this object is not Trainer1')

class MemoryAndHierarchicalTPG1(TPG):

    def __init__(self, actions=2, teamPopSize: int = 1000, rootBasedPop: bool = True, gap: float = 0.5, inputSize: int = 33600, nRegisters: int = 8, initMaxTeamSize: int = 10, initMaxProgSize: int = 10, maxTeamSize: int = -1, pLrnDel: float = 0.7, pLrnAdd: float = 0.6, pLrnMut: float = 0.2, pProgMut: float = 0.1, pActMut: float = 0.1, pActAtom: float = 0.95, pInstDel: float = 0.5, pInstAdd: float = 0.4, pInstSwp: float = 0.2, pInstMut: float = 1.0, doElites: bool = True, memType="def", memMatrixShape: tuple = (100, 8), rampancy: tuple = (0, 0, 0), operationSet: str = "custom", traversal: str = "team", prevPops=None, mutatePrevs=True, initMaxActProgSize: int = 6, nActRegisters: int = 4):
        self.trainer = Trainer3(actions, teamPopSize, rootBasedPop, gap, inputSize, nRegisters, initMaxTeamSize, initMaxProgSize, maxTeamSize, pLrnDel, pLrnAdd, pLrnMut, pProgMut, pActMut, pActAtom, pInstDel, pInstAdd, pInstSwp, pInstMut, doElites, memType, memMatrixShape, rampancy, operationSet, traversal, prevPops, mutatePrevs, initMaxActProgSize, nActRegisters)

    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, Trainer3): raise Exception('this object is not Trainer3')

class EmulatorTPG(TPG):
    def __init__(self, 
        actions=2,
        state=np.arange(4, dtype=float),
        teamPopSize:int=1000,               # *
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
        memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4,
        thinkingTime:int=2
    ):
        # super().__init__()
        self.ActorGraph = Trainer3(
            actions, 
            teamPopSize,               # *
            rootBasedPop,             
            gap,                      
            inputSize,                
            nRegisters,                   # *
            initMaxTeamSize,             # *
            initMaxProgSize,             # *
            maxTeamSize,                 # *
            pLrnDel,                  # *
            pLrnAdd,                  # *
            pLrnMut,                  # *
            pProgMut,                 # *
            pActMut,                  # *
            pActAtom,                # *
            pInstDel,                 # *
            pInstAdd,                 # *
            pInstSwp,                 # *
            pInstMut,                 # *
            doElites, 
            memType, 
            memMatrixShape,       # *
            rampancy,
            operationSet, 
            traversal, 
            prevPops, mutatePrevs,
            initMaxActProgSize,           # *
            nActRegisters,
        )
        self.EmulatorGraph = Trainer2(
            state,
            teamPopSize,               # *
            rootBasedPop,             
            gap,                      
            inputSize,                
            nRegisters,                   # *
            initMaxTeamSize,             # *
            initMaxProgSize,             # *
            maxTeamSize,                 # *
            pLrnDel,                  # *
            pLrnAdd,                  # *
            pLrnMut,                  # *
            pProgMut,                 # *
            pActMut,                  # *
            pActAtom,                # *
            pInstDel,                 # *
            pInstAdd,                 # *
            pInstSwp,                 # *
            pInstMut,                 # *
            doElites, 
            memType, 
            memMatrixShape,       # *
            rampancy,
            operationSet, 
            traversal, 
            prevPops, mutatePrevs,
            initMaxActProgSize,           # *
            nActRegisters,
        )
        self.thinkingTimeLimit=thinkingTime
        self.initParams()

    def instance_valid(self, _actor, _emulator) -> bool:
        if not isinstance(_actor, Trainer3): raise Exception('this actor is not Trainer1')
        if not isinstance(_emulator, Trainer2): raise Exception('this emulator is not Trainer2')

    def _setupParams(self, actor_id, emulator_id=None, actions=None, images=None, reward=None):
        self.actions[actor_id]=[]
        if not self.rewards.get(actor_id) : self.rewards[actor_id]=0.
        if emulator_id is not None:
            self.pairs[actor_id] = emulator_id
            if not self.memories.get(emulator_id) : self.memories[emulator_id]=[]
            if images is not None: self.memories[emulator_id]   += images
        if actions is not None: self.actions[actor_id]      = actions
        if reward is not None: self.rewards[actor_id]       += reward
 
    def initParams(self):
        self.actions    = {}
        self.memories   = {}
        self.pairs      = {}
        self.rewards    = {}
    
    def think(self, hippocampus, _actor, _emulator):
        assert isinstance(_actor, Agent3)
        assert isinstance(_emulator, Agent2)
        _actor.configFunctionsSelf()
        _emulator.configFunctionsSelf()
        state = np.array(hippocampus['now'])
        actor_id = str(_actor.team.id)
        emulator_id = str(_emulator.team.id)
        ActionObject3.actions = hippocampus['actions']
        MemoryObject.memories = hippocampus['memories']
        actionCodes = []
        memoryCodes = []
        rewards = 0
        timeout_start = time.time()
        while time.time() < timeout_start + self.thinkingTimeLimit:
            actionCode = _actor.act(state)
            imageCode, reward = _emulator.image(actionCode, state)
            actionCodes  += [actionCode]
            rewards      += reward*0.001
            memoryCodes  += [imageCode]
            state = hippocampus['memories'][imageCode].recall(state)
            # breakpoint(self.thinkingTimeLimit)
        return actor_id, emulator_id, actionCodes, memoryCodes, rewards 

    def thinker(self, _state, _actors, _emulators, actionQ):
        print()
        manager = mp.Manager()
        hippocampus = manager.dict()
        hippocampus['actions']  = ActionObject3.actions
        hippocampus['memories'] = MemoryObject.memories
        hippocampus['now'] = _state
        # breakpoint(hippocampus['memories'].choice())
        with mp.Pool(mp.cpu_count()-2) as pool:
            results = pool.starmap(self.think, [(hippocampus, actor, emulator) for actor, emulator in zip(_actors, _emulators)])
        # breakpoint(results)s
        for result in results:
            actor_id, emulator_id, actions, images, reward = result
            self._setupParams(actor_id=actor_id, emulator_id=emulator_id, actions=actions, images=images, reward=reward)
        bestActor=max(self.rewards, key=lambda k: self.rewards.get(k))
        self.best=bestActor

        return self.actions[bestActor]
    
    def actor(self, _init_state, _elite):
        return [_elite.act(_init_state)]


    def episode(self, _actors, _emulators, _env, _elite_actor, _elite_emulator, _logger=None, _scores={}, _frames:int=500, _show=False):
        _scores = {}
        _states = {}
        frame=0
        actionQ = queue.Queue()
        executor = ThreadPoolExecutor(max_workers=3)
        action_list = []
        state = _env.reset().flatten() # get initial state and prep environment
        thinker = executor.submit(self.thinker, state, _actors, _emulators)
        actor = executor.submit(self.actor, state, _elite_actor, _elite_emulator)
        action_list.append(thinker)
        action_list.append(actor)
        while frame<_frames:
            score = 0
            states = []
            # action_list = await asyncio.gather(
            #     self.thinker(state, _actors, _emulators),
            #     self.actor(state, _elite)
            # ) 
            for actions in action_list: # run episodes that last 500 frames
                for act in actions.result():
                    frame+=1
                    if not act in range(_env.action_space.n): continue
                    state, reward, isDone, debug = _env.step(act) # state : np.ndarray.flatten
                    score+=reward # accumulate reward in score
                    states.append(state.flatten())
                    if isDone: 
                        frame=_frames
                        break
                        # end early if losing state
                    if _show:  self.show_state(_env, frame)
            
            if _scores.get(self.best)    is None : _scores[self.best]=0
            if _states.get(self.pairs[self.best]) is None : _states[self.pairs[self.best]]=[]
            _scores[self.best]    += score      # store score
            _states[self.pairs[self.best]] += states  # store states
            action_list = self.thinker(state, _actors, _emulators)
            # else:
            #     self.best = str(_elite.team.id)
            #     self._setupParams(self.best)
            #     act = _elite.act(state)
            #     frame+=1
            #     if not act in range(_env.action_space.n): continue
            #     state, reward, isDone, debug = _env.step(act) # state : np.ndarray.flatten
            #     states.append(state.flatten())
            #     if _scores.get(self.best)    is None : _scores[self.best]=0
            #     _scores[self.best]    += reward      # store score
            #     if isDone: break
            #     if _show:  self.show_state(_env, frame)


        if _logger is not None: _logger.info(f'{self.best},{score}')

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores, _states

    def generation(self,_actor, _emulator, _env, _logger=None, _episodes=1, _frames= 500, _show=False):

        _scores = {}
        _states = {}
        _task       = _env.spec.id
        actors      = _actor.getAgents(task=_task)
        elite       = _actor.getEliteAgent(task=_task)
        emulators   = _emulator.getAgents(task=_task)

        # breakpoint(type(emulators))
        for _ in range(_episodes):      _scores, _states = self.episode(_actors=actors, _emulators=emulators, _env=_env, _elite=elite, _logger=_logger, _scores=_scores, _frames=_frames, _show=_show)
        for ac in _scores:               
            _scores[ac]/=_episodes
            _scores[ac] = tanh(_scores[ac])
            _scores[ac]-=tanh(self.rewards[ac])
        # emulator 用の二乗和誤差平均の計算
        states=[]
        for em in _states:
            score=0
            for state, imageCode in zip(_states[em], self.memories[em]):
                
                diff = MemoryObject.memories[imageCode].memorize(state)
                score += np.power(diff, 2).sum()*0.01
                states+=[state]
            _states[em]=score
        
        # 報酬の贈与
        for actor in actors:
            if _scores.get(str(actor.team.id)) : 
                actor.reward(_scores[str(actor.team.id)],task=_task)
        # ここらへんのエミュレータの報酬設計
        for emulator in emulators:
            if _states.get(str(emulator.team.id)): 
                emulator.reward(_states[str(emulator.team.id)],task=_task)
        # breakpoint()
        _actor.evolve([_task])
        # breakpoint('finish')
        _emulator.evolve([_task], _states=states)

        self.initParams()

        return _scores
    
    def growing(self, _actor, _emulator, _task:str, _generations:int=1000, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True):
        self.instance_valid(_actor, _emulator)
        logger, filename = setup_logger(__name__, _task, test=_test, load=_load)
        env = gym.make(_task) # make the environment
        action_space = env.action_space
        observation_space = env.observation_space
        
        action = 0
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n

        state = observation_space.sample()
        _actor.resetActions(actions=action)
        _emulator.resetMemories(state=state.flatten())
        def outHandler(signum, frame):
            if not _test: 
                _actor.saveToFile(f'{_task}/{filename}-act')
                _emulator.saveToFile(f'{_task}/{filename}-emu')
            print('exit')
            sys.exit()
        
        signal.signal(signal.SIGINT, outHandler)

        summaryScores = []

        tStart = time.time()
        for gen in tqdm(range(_generations)): # generation loop
            # breakpoint(type(_emulator))
            scores = self.generation(_actor=_actor, _emulator=_emulator, _env=env, _logger=logger, _episodes=_episodes, _frames=_frames, _show=_show)

            score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            logger.info(f'generation:{gen}, score:{score}')
            summaryScores.append(score)

        #clear_output(wait=True)
        logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
        logger.info(f'Results: Min, Max, Avg, {summaryScores}')
        return filename

    def start(self, _task, _show, _test, _load, _actor=None, _emulator=None, _generations=1000, _episodes=1, _frames=500):
        if _actor is None: 
            if not self.ActorGraph : raise Exception('actor is not defined')
            _actor = self.ActorGraph
        if _emulator is None: 
            if not self.EmulatorGraph : raise Exception('emulator is not defined')
            _emulator = self.EmulatorGraph

        _filename = self.growing(_actor=_actor, _emulator=_emulator, _task=_task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load)
        if not _test: 
            _actor.saveToFile(f'{task}/{_filename}-act')
            _emulator.saveToFile(f'{task}/{_filename}-emu')
        return _filename

if __name__ == '__main__':
    task = sys.argv[1]
    show = False
    test = False
    load = False
    actor = None
    emulator = None
    teamPopSize=10
    generations=1000
    episodes=1
    frames=1000
    thinkingTime=0.2

    tpg = NativeTPG(teamPopSize=teamPopSize)


    for arg in sys.argv[2:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
        if 'teamPopSize:' in arg: teamPopSize=int(arg.split(':')[1])
        if 'generatioins:' in arg: generations=int(arg.split(':')[1])
        if 'episodes:' in arg: episodes=int(arg.split(':')[1])
        if 'frames:' in arg: frames=int(arg.split(':')[1])
        if 'thinkingTime:' in arg: thinkingTime=float(arg.split(':')[1])
    
    for arg in sys.argv[2:]:
        if arg=='native':
            tpg = NativeTPG(teamPopSize=teamPopSize)
            print('native')
        if arg=='hierarchy':
            tpg = MemoryAndHierarchicalTPG(teamPopSize=teamPopSize)
            print('hierarchy')
        if arg=='hierarchy1':
            tpg = MemoryAndHierarchicalTPG1(teamPopSize=teamPopSize)
            print('hierarchy1')
        if arg=='emulate':
            tpg = EmulatorTPG(teamPopSize=teamPopSize, thinkingTime=thinkingTime)
            print('emulate')
        
        if 'actor:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            actor = loadTrainer(modelPath)
        if 'emulator:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            emulator = loadTrainer(modelPath)

    if isinstance(tpg, EmulatorTPG):
        tpg.start(_task=task, _show=show, _test=test, _load=load, _actor=actor, _emulator=emulator, _generations=generations, _episodes=episodes, _frames=frames)
    else:
        tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=actor, _generations=generations, _episodes=episodes, _frames=frames)
