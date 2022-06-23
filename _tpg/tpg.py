from _tpg.trainer import Trainer2
import asyncio

class EmulatorTPG:
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
        thinkingTime:int=5
    ):
        self.ActorGraph = Trainer2(
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
        self.actions = {}
        self.states = {}
        self.rewards = {}
        self.thinkingTimeLimit=thinkingTime

    async def actor(self, _state):
        while True:
            for agent in self.ActorGraph.getAgents():
                yield agent.act(_state), str(agent.team.id)

    async def emulator(self, _act, _prestate):
        while True:
            for agent in self.EmulatorGraph.getAgents():
                yield agent.image(_act, _prestate), str(agent.team.id)

    async def think(self, _state):
        act, actor_id = await self.actor(_state)
        state, reward, emulator_id = await self.emulator(act, _state)

    async def thinker(self, _state):
        try:
            await asyncio.wait_for(self.think, timeout=self.thinkingTimeLimit)
        except asyncio.TimeoutError:
            pass
