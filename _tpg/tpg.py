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
        # logger.shutdown()
        # logging.disable(logging.NOTSET)

        list(map(logger.removeHandler, logger.handlers))
        list(map(logger.removeFilter, logger.filters))
        return filename

    def start(self, _task, _show, _test, _load, _trainer=None, _generations=1000, _episodes=1, _frames=500):
        if _trainer is None : 
            if not self.trainer : raise Exception('trainer is not defined')
            _trainer = self.trainer

        _filename = self.growing(_trainer, _task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load)
        if not _test: _trainer.saveToFile(f'{_task}/{_filename}')
        return _filename
    
    def instance_valid(self, trainer)->bool: pass

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

class StateTPG(TPG):

    def __init__(self, state=np.array([1.]), teamPopSize: int = 1000, rootBasedPop: bool = True, gap: float = 0.5, inputSize: int = 33600, nRegisters: int = 8, initMaxTeamSize: int = 10, initMaxProgSize: int = 10, maxTeamSize: int = -1, pLrnDel: float = 0.7, pLrnAdd: float = 0.6, pLrnMut: float = 0.2, pProgMut: float = 0.1, pActMut: float = 0.1, pActAtom: float = 0.95, pInstDel: float = 0.5, pInstAdd: float = 0.4, pInstSwp: float = 0.2, pInstMut: float = 1.0, doElites: bool = True, memType="def", memMatrixShape: tuple = (100, 8), rampancy: tuple = (0, 0, 0), operationSet: str = "custom", traversal: str = "team", prevPops=None, mutatePrevs=True, initMaxActProgSize: int = 6, nActRegisters: int = 4):
        self.trainer = Trainer2(state, teamPopSize, rootBasedPop, gap, inputSize, nRegisters, initMaxTeamSize, initMaxProgSize, maxTeamSize, pLrnDel, pLrnAdd, pLrnMut, pProgMut, pActMut, pActAtom, pInstDel, pInstAdd, pInstSwp, pInstMut, doElites, memType, memMatrixShape, rampancy, operationSet, traversal, prevPops, mutatePrevs, initMaxActProgSize, nActRegisters)

    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, Trainer2): raise Exception('this object is not Trainer2')
    
    def episode(self,_agents, _env, _logger=None, _scores={}, _frames:int=500, _show=False):
        
        states = []
        unexpectancies = []
        for agent in _agents: # to multi-proccess
            
            state = _env.reset() # get initial state and prep environment
            score = 0
            _id = str(agent.team.id)
            for _ in range(_frames): # run episodes that last 500 frames
                act = _env.action_space.sample()
                imageCode = agent.image(act, state.flatten())
                state, reward, isDone, debug = _env.step(act)
                diff, unex = MemoryObject.memories[imageCode].memorize(state.flatten(), reward)

                score += tanh(np.power(diff, 2).sum())
                states+=[state.flatten()]
                unexpectancies+=[unex]

                if isDone: break # end early if losing state
                if _show:
                    self.show_state(_env, _)

            if _scores.get(_id) is None : _scores[_id]=0
            _scores[_id] += score # store score

            if _logger is not None: _logger.info(f'{_id},{score}')

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores, states, unexpectancies

    def generation(self,_trainer, _env, _logger=None, _episodes=1, _frames= 500, _show=False):
        _scores = {}
        agents = _trainer.getAgents()
        _task = _env.spec.id
        for _ in range(_episodes):      _scores, states, unexpectancies = self.episode(agents, _env, _logger=_logger, _scores=_scores, _frames=_frames, _show=_show)
        for i in _scores:                _scores[i]/=_episodes
        for agent in agents:            agent.reward(_scores[str(agent.team.id)],task=_task)
        _trainer.evolve([_task], _states=states, _unexpectancies=unexpectancies)

        return _scores
    
    def growing(self, _trainer, _task:str, _generations:int=1000, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True):
        self.instance_valid(_trainer)
        logger, filename = setup_logger(__name__, _task, test=_test, load=_load)
        env = gym.make(_task) # make the environment

        _trainer.resetMemories(state=env.observation_space.sample().flatten())

        def outHandler(signum, frame):
            if not _test: _trainer.saveToFile(f'{_task}/{filename}-em')
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
            
        logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
        logger.info(f'Results: Min, Max, Avg, {summaryScores}')


        list(map(logger.removeHandler, logger.handlers))
        list(map(logger.removeFilter, logger.filters))
        
        return filename
    
class EmulatorTPG(TPG):
    state=np.array([],dtype=float)
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

    def _setupParams(self, actor_id, emulator_id, actions=None, images=None, rewards=None):
        self.actions[actor_id]=[]
        self.rewards[actor_id]=0.
        if rewards is not None: self.rewards[actor_id] = sum(rewards)
        if actions is not None: self.actions[actor_id] = actions

        self.pairs[actor_id] = emulator_id
        if not self.memories.get(emulator_id) : self.memories[emulator_id]=[]
        if images is not None: self.memories[emulator_id]   += images

 
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
        rewards     = []
        timeout_start = time.time()
        while time.time() < timeout_start + self.thinkingTimeLimit:
            actionCode = _actor.act(state)
            imageCode  = _emulator.image(actionCode, state)
            actionCodes  += [actionCode]
            memoryCodes  += [imageCode]
            rewards += [MemoryObject.memories[imageCode].reward]
            state = hippocampus['memories'][imageCode].recall(state)
            # breakpoint(self.thinkingTimeLimit)
        return actor_id, emulator_id, actionCodes, memoryCodes, rewards

    def thinker(self, _state, _actors, _emulators):
        manager = mp.Manager()
        hippocampus = manager.dict()
        hippocampus['actions']  = ActionObject3.actions
        hippocampus['memories'] = MemoryObject.memories
        hippocampus['now'] = _state
        with mp.Pool(mp.cpu_count()-2) as pool:
            # params = [(actor, emulator) for actor, emulator in zip(_actors, _emulators)]
            results = pool.starmap(self.think, [(hippocampus, actor, emulator) for actor, emulator in zip(_actors, _emulators)])
        for result in results:
            actor_id, emulator_id, actions, images, rewards = result
            self._setupParams(actor_id=actor_id, emulator_id=emulator_id, actions=actions, images=images, rewards=rewards)
            assert self.pairs.get(actor_id), (self.pairs[actor_id], emulator_id)
        bestActor=max(self.rewards, key=lambda k: self.rewards.get(k))

        return bestActor, self.pairs[bestActor]

    def episode(self, _actors, _emulators, _env, _elite_actor, _logger=None, _scores={}, _frames:int=500, _show=False):
        _scores = {}
        _states = {}
        frame=0
        executor = ThreadPoolExecutor(max_workers=2)
        # state = _env.reset() # get initial state and prep environment
        thinker = executor.submit(self.thinker, EmulatorTPG.state.flatten(), _actors, _emulators)
        # self.best = str(_elite_actor.team.id)
        total_reward = 0.
        # thinking_actor = []
        while frame<_frames:
            if thinker.done():
                scores = []
                states = []
                # thinking_actor.append(self.best)
                bestActor, pairEmulator = thinker.result()
                for actionCode in self.actions[bestActor]:
                    frame+=1
                    if not ActionObject3.actions[actionCode] in range(_env.action_space.n): continue
                    state, reward, isDone, debug = _env.step(ActionObject3.actions[actionCode]) # state : np.ndarray.flatten
                    scores+=[reward] # accumulate reward in score
                    total_reward += reward
                    # print(reward)
                    states.append(state.flatten())
                    if isDone: 
                        frame=_frames
                        EmulatorTPG.state=_env.reset()
                        break
                        # end early if losing state
                    if _show:  self.show_state(_env, frame)
                    EmulatorTPG.state = state
                
                # breakpoint('thinker ok')
                if _scores.get(bestActor)    is None : _scores[bestActor]=[]
                if _states.get(pairEmulator) is None : 
                    assert pairEmulator, pairEmulator
                    _states[pairEmulator]=[]
                _scores[bestActor]    += scores             # store score
                _states[pairEmulator] += states    # store states
                thinker = executor.submit(self.thinker, EmulatorTPG.state.flatten(), _actors, _emulators)
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

    def generation(self,_actor, _emulator, _env, _logger=None, _episodes=1, _frames= 500, _show=False):

        _scores = {}
        _states = {}
        _task       = _env.spec.id
        actors      = _actor.getAgents(task=_task)
        elite_actor = _actor.getEliteAgent(task=_task)
        emulators   = _emulator.getAgents(task=_task)

        # breakpoint(type(emulators))
        for _ in range(_episodes):
            _scores, _states, total_reward = self.episode(
                _actors=actors, 
                _emulators=emulators, 
                _env=_env,
                _elite_actor=elite_actor, 
                _logger=_logger, 
                _scores=_scores, 
                _frames=_frames, 
                _show=_show
            )

        reward_for_actor = {}
        states=[]

        unexpectancy=0.
        unexpectancies=[]
        for ac in _scores:               
            total = sum(_scores[ac])
            total = tanh(total)
            total-= tanh(self.rewards[ac])
            reward_for_actor[ac] = total
            em = self.pairs[ac]
            score = 0.
            assert len(_states[em])==len(_scores[ac])
            # unexpectancyが予想よりいいか悪いかを表す。
            unexpectancy = abs(total)

            for state, imageCode, reward in zip(_states[em], self.memories[em], _scores[ac]):
                diff, unex = MemoryObject.memories[imageCode].memorize(state, reward)
                # print(reward)
                score += tanh(np.power(diff, 2).sum())
                states+=[state]
                unexpectancies+=[unex]
            if unexpectancy==0: _states[em]=tanh(score)*1000
            else: _states[em]=tanh(score)/unexpectancy 
            # 予想外度　-1~1ぐらい、予想外度が小さいとあまりスコアを得られない。
            # ただ、エミュレータの場合、予想外度が小さいものほど評価されるべき。
            # 報酬予想が正しいものが残る。
            # 報酬予測が予想外のものほど、短期的に、銘記されやすい。
            # pop up されやすい。
            # 逆に、予想外のチームはスコアは得られにくい。

                
        # 報酬の贈与
        for actor in actors:
            if reward_for_actor.get(str(actor.team.id)) : 
                actor.reward(reward_for_actor[str(actor.team.id)],task=_task)
        # ここらへんのエミュレータの報酬設計
        for emulator in emulators:
            if _states.get(str(emulator.team.id)): 
                emulator.reward(_states[str(emulator.team.id)],task=_task)
        # breakpoint()
        _actor.evolve([_task])
        # breakpoint('finish')
        _emulator.evolve([_task], _states=states, _unexpectancies=unexpectancies)
        # breakpoint('after evolving')
        # print(unexpectancies)

        self.initParams()

        return total_reward
    
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
        EmulatorTPG.state = env.reset()
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
            total_score = self.generation(_actor=_actor, _emulator=_emulator, _env=env, _logger=logger, _episodes=_episodes, _frames=_frames, _show=_show)

            # score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            logger.info(f'generation:{gen}, score:{total_score}')
            summaryScores.append(total_score)

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

class Automata(TPG):
    def __init__(self, actions=2, teamPopSize: int = 1000, rootBasedPop: bool = True, gap: float = 0.5, inputSize: int = 33600, nRegisters: int = 8, initMaxTeamSize: int = 10, initMaxProgSize: int = 10, maxTeamSize: int = -1, pLrnDel: float = 0.7, pLrnAdd: float = 0.6, pLrnMut: float = 0.2, pProgMut: float = 0.1, pActMut: float = 0.1, pActAtom: float = 0.95, pInstDel: float = 0.5, pInstAdd: float = 0.4, pInstSwp: float = 0.2, pInstMut: float = 1., doElites: bool = True, memType="def", memMatrixShape: tuple = (100, 8), rampancy: tuple = (0, 0, 0), operationSet: str = "custom", traversal: str = "team", prevPops=None, mutatePrevs=True, initMaxActProgSize: int = 6, nActRegisters: int = 4):
        super().__init__(actions, teamPopSize, rootBasedPop, gap, inputSize, nRegisters, initMaxTeamSize, initMaxProgSize, maxTeamSize, pLrnDel, pLrnAdd, pLrnMut, pProgMut, pActMut, pActAtom, pInstDel, pInstAdd, pInstSwp, pInstMut, doElites, memType, memMatrixShape, rampancy, operationSet, traversal, prevPops, mutatePrevs, initMaxActProgSize, nActRegisters)

if __name__ == '__main__':
    # task = sys.argv[1]
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


    for arg in sys.argv[1:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
        if 'task:' in arg: task=arg.split(':')[1]
        if 'teamPopSize:' in arg: teamPopSize=int(arg.split(':')[1])
        if 'generations:' in arg: generations=int(arg.split(':')[1])
        if 'episodes:' in arg: episodes=int(arg.split(':')[1])
        if 'frames:' in arg: frames=int(arg.split(':')[1])
        if 'thinkingTime:' in arg: thinkingTime=float(arg.split(':')[1])
    
    for arg in sys.argv[1:]:
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
        if arg=='state':
            tpg = StateTPG(teamPopSize=teamPopSize)
            print('state')
        
        if 'actor:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            actor = loadTrainer(modelPath)
        if 'emulator:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            emulator = loadTrainer(modelPath)

    if not task: raise Exception("task doesn't")

    if isinstance(tpg, EmulatorTPG):
        tpg.start(_task=task, _show=show, _test=test, _load=load, _actor=actor, _emulator=emulator, _generations=generations, _episodes=episodes, _frames=frames)
    else:
        tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=actor, _generations=generations, _episodes=episodes, _frames=frames)
