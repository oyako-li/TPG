import asyncio
import matplotlib.pyplot as plt
# from IPython import display
import sys
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

    # To transform pixel matrix to a single vector.
    def getState(self,inState):
        # each row is all 1 color
        rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

        # add each with appropriate shifting
        # get RRRRRRRR GGGGGGGG BBBBBBBB
        return np.add(np.left_shift(rgbRows[0], 16),
            np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))


    # 5 generations isn't much (not even close), but some improvements
    # should be seen.
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
        print(filename)
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
        if not _test: _trainer.saveToFile(f'{task}/{_filename}')
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

class EmulatorTPG(TPG):
    def __init__(self, 
        actions=2,
        state=4,
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
        thinkingTime:int=5
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
        self.rewards = {}
        self.thinkingTimeLimit=thinkingTime

    def getActor(self):
        yield from self.ActorGraph.getAgents()

    def getEmulator(self):
        yield from self.EmulatorGraph.getAgents()

    async def think(self, _state):
        self.actions = {[]}
        self.states = {[]}
        self.rewards = {}
        self.actor = self.getActor()
        self.emulator = self.getEmulator()
        while True:
            act = self.actor.act(_state)
            if act == 'break': break
            _state, reward = self.emulator.image(act, _state)
            self.states[self.emulator.team.id]+=[_state]
            self.actions[self.actor.team.id]+=[act]
            self.rewards[self.actor.team.id]+=reward

    async def thinker(self, _state):
        try:
            await asyncio.wait_for(self.think(_state), timeout=self.thinkingTimeLimit)
        except asyncio.TimeoutError:
            pass

        bestActions=max(self.rewards, key=lambda k: self.rewards.get(k))
        return self.actions[bestActions]


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
        if arg=='emulator':
            tpg = EmulatorTPG(teamPopSize=teamPopSize)
        
        if 'model:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            trainer = loadTrainer(modelPath)

    tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=trainer, _generations=generations, _episodes=episodes, _frames=frames)
