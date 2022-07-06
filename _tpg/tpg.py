import asyncio
import matplotlib.pyplot as plt
# from IPython import display
import sys
from _tpg.memory_object import MemoryObject
# sys.path.insert(0, '.')

import gym
from _tpg.trainer import Trainer, Trainer1, Trainer11, Trainer2, loadTrainer
from _tpg.base_log import setup_logger
from tqdm import tqdm
import signal
import time
import numpy as np

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
                    self.show_state(
                        _env, _
                    )

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
        self.trainer = Trainer11(actions, teamPopSize, rootBasedPop, gap, inputSize, nRegisters, initMaxTeamSize, initMaxProgSize, maxTeamSize, pLrnDel, pLrnAdd, pLrnMut, pProgMut, pActMut, pActAtom, pInstDel, pInstAdd, pInstSwp, pInstMut, doElites, memType, memMatrixShape, rampancy, operationSet, traversal, prevPops, mutatePrevs, initMaxActProgSize, nActRegisters)

    def instance_valid(self, trainer) -> bool:
        if not isinstance(trainer, Trainer11): raise Exception('this object is not Trainer11')

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
        self.ActorGraph = Trainer11(
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
        self.actions = {[]}
        self.states = {[]}
        self.memories = {[]}
        self.pair = {}
        self.rewards = {}
        self.thinkingTimeLimit=thinkingTime

    def instance_valid(self, _actor, _emulator) -> bool:
        if not isinstance(_actor, Trainer11): raise Exception('this actor is not Trainer11')
        if not isinstance(_emulator, Trainer2): raise Exception('this emulator is not Trainer2')

    def getActor(self):
        yield from self.ActorGraph.getAgents()

    def getEmulator(self):
        yield from self.EmulatorGraph.getAgents()

    async def counterfactualThinking(self, _state, _actor, _emulator):

        actor_id = str(_actor.team.id)
        emulator_id = str(_emulator.team.id)
        self.actions[actor_id]=[]
        self.rewards[actor_id]=0.
        self.pair[actor_id] = emulator_id

        if self.memories.get(emulator_id) is None : self.memories[emulator_id]=[]
        if self.states.get(emulator_id)   is None : self.states[emulator_id]  =[]

        while True:
            act = _actor.act(_state)
            self.actions[actor_id]      += [act]
            if act == 'break': break
            _state, imageCode, reward = _emulator.image(act, _state)
            self.rewards[actor_id]      += reward
            self.memories[emulator_id]  += [imageCode]
            self.states[emulator_id]    += [_state]
    
    async def think(self, _state, _actors, _emulators):
        # combination = [(actor, emulator) for actor in _actors]
        cors = [self.counterfactualThinking(_state, actor, emulator) for actor, emulator in zip(_actors, _emulators)]
        await asyncio.gather(*cors)

    async def thinker(self, _state, _actors, _emulators):
        try:
            await asyncio.wait_for(self.think(_state, _actors, _emulators), timeout=self.thinkingTimeLimit)
        except asyncio.TimeoutError:
            pass

        bestActor=max(self.rewards, key=lambda k: self.rewards.get(k))
        pairEmulator = self.pair[bestActor]
        return bestActor, pairEmulator

    async def episode(self, _actors, _emulators, _env, _logger=None, _scores={}, _frames:int=500, _show=False):
        _scores = {}
        _states = {[]}
        frame=0
        state = _env.reset() # get initial state and prep environment
        while frame<_frames:
            score = 0
            states = []
            bestActor, pairEmulator = await self.thinker(state.flatten(), _actors, _emulators)
            for act in self.actions[bestActor]: # run episodes that last 500 frames
                if not act in range(_env.action_space.n): continue
                state, reward, isDone, debug = _env.step(act) # state : np.ndarray.flatten
                score+=reward # accumulate reward in score
                states.append(state.flatten())
                frame+=1
                if isDone: break # end early if losing state
                if _show:  self.show_state(_env, frame)

            if _scores.get(bestActor)    is None : _scores[bestActor]=0
            if _states.get(pairEmulator) is None : _states[pairEmulator]=[]
            _scores[bestActor]    += score      # store score
            _states[pairEmulator] += [states]  # store states

            if _logger is not None: _logger.info(f'{bestActor},{score}')

        # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

        return _scores, _states

    async def generation(self,_actor, _emulator, _env, _logger=None, _episodes=1, _frames= 500, _show=False):
        _scores = {}
        actors      = _actor.getAgents()
        emulators   = _emulator.getAgents()
        _task       = _env.spec.id
        for _ in range(_episodes):      _scores, _states = await self.episode(actors, emulators, _env, _logger=_logger, _scores=_scores, _frames=_frames, _show=_show)
        for i in _scores:               _scores[i]/=_episodes
        # emulator 用の二乗和誤差平均の計算
        last_state = _states[-1]

        for em in _states:
            score=0
            for state, imageCode in zip(_states[em], self.memories[em]):

                key = MemoryObject.memories[imageCode].keys()
                val = MemoryObject.memories[imageCode].values()
                diff = val-state[key]
                val = val-(diff)*0.01 # 学習率
                MemoryObject.memories[imageCode].update(val)
                score += np.power(diff, 2).sum()
            _states[em]=score
        
        # 報酬の贈与
        for actor in actors:
            actor.reward(_scores[str(actor.team.id)],task=_task)
        # ここらへんのエミュレータの報酬設計
        for emulator in emulators:
            emulator.reward(_states[str(emulator.team.id)],task=_task)
        _actor.evolve([_task])
        _emulator.evolve([_task], _state=last_state)
        return _scores
    
    async def growing(self, _actor, _emulator, _task:str, _generations:int=1000, _episodes:int=1, _frames:int=500, _show=False, _test=False, _load=True):
        self.instance_valid(_actor, _emulator)
        logger, filename = setup_logger(__name__, _task, test=_test, load=_load)
        print(_task,filename)
        env = gym.make(_task) # make the environment
        action_space = env.action_space
        # obsesrvation_space = env.observation_space
        
        action = 0
        if isinstance(action_space, gym.spaces.Box):
            action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
        elif isinstance(action_space, gym.spaces.Discrete):
            action = action_space.n
        _actor.resetActions(actions=action)

        state = env.observation_space.sample()
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
            scores = await self.generation(_actor, _emulator, env, logger, _episodes=_episodes, _frames=_frames, _show=_show)

            score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

            logger.info(f'generation:{gen}, score:{score}')
            summaryScores.append(score)

        #clear_output(wait=True)
        logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
        logger.info(f'Results: Min, Max, Avg, {summaryScores}')
        return filename

    async def start(self, _task, _show, _test, _load, _actor=None, _emulator=None, _generations=1000, _episodes=1, _frames=500):
        if _actor is None: 
            if not self.ActorGraph : raise Exception('actor is not defined')
            _actor = self.ActorGraph
        if _emulator is None: 
            if not self.EmulatorGraph : raise Exception('emulator is not defined')
            _emulator = self.EmulatorGraph

        _filename = await self.growing(_actor, _emulator, _task, _generations=_generations, _episodes=_episodes, _frames=_frames, _show=_show, _test=_test, _load=_load)
        if not _test: 
            _actor.saveToFile(f'{task}/{_filename}-act')
            _emulator.saveToFile(f'{task}/{_filename}-emu')
        return _filename

if __name__ == '__main__':
    task = sys.argv[1]
    show = False
    test = False
    load = False
    trainer = None
    teamPopSize=10
    generations=1000
    episodes=1
    frames=1000

    tpg = NativeTPG(teamPopSize=teamPopSize)


    for arg in sys.argv[2:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
        if 'teamPopSize:' in arg: teamPopSize=int(arg.split(':')[1])
        if 'generatioins:' in arg: generations=int(arg.split(':')[1])
        if 'episodes:' in arg: episodes=int(arg.split(':')[1])
        if 'frames:' in arg: frames=int(arg.split(':')[1])
    
    for arg in sys.argv[2:]:
        if arg=='native':
            tpg = NativeTPG(teamPopSize=teamPopSize)
        if arg=='hierarchy':
            tpg = MemoryAndHierarchicalTPG(teamPopSize=teamPopSize)
        if arg=='hierarchy1':
            tpg = MemoryAndHierarchicalTPG1(teamPopSize=teamPopSize)
        if arg=='emulator':
            tpg = EmulatorTPG(teamPopSize=teamPopSize)
        
        if 'model:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            trainer = loadTrainer(modelPath)
    if isinstance(tpg, EmulatorTPG): asyncio.run(tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=trainer, _generations=generations, _episodes=episodes, _frames=frames))
    else:    tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=trainer, _generations=generations, _episodes=episodes, _frames=frames)
