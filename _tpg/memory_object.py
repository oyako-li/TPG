import numpy as np
import random
from _tpg.utils import flip

class MemoryObject:
    memories=[{}]
    def __init__(self, _state=0, initParam=None):
        from _tpg.team import Team2

        if isinstance(_state, Team2):
            self.teamMemory = _state
            self.memoryCode = None
            return
        elif isinstance(_state, MemoryObject):
            self.memoryCode = _state.memoryCode
            self.teamMemory = _state.teamMemory
            return
        elif isinstance(_state, int):
            # if _state > len(MemoryObject._memorys)-1: raise IndexError
            _state %= len(MemoryObject.memories)
            self.memoryCode = _state
            self.teamMemory = None
            return
        elif isinstance(_state, np.ndarray):
            if initParam is not None:
                if "stateRandomMemorizeTimes" not in initParam: raise Exception('stateRandomMemorizeTimes not found in init params', initParam)
                memory={}
                for _ in initParam['stateRandomMemorizeTimes']:
                    key = random.randint(0,len(_state)-1)
                    memory[key]=_state[key]
                MemoryObject.memories.append(memory)
                self.memoryCode = len(MemoryObject.memories)-1
                self.teamMemory = None

            else:
                memory = dict(zip(range(len(_state)), _state))
                MemoryObject.memories.append(memory)
                self.memoryCode = len(MemoryObject.memories)-1
                self.teamMemory = None

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, MemoryObject) and not isinstance(__o, np.ndarray): False
        memory:dict = MemoryObject.memories[self.memoryCode]

        for i in memory:
            if __o[i]!=memory[i]: False

        return True

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __getitem__(self, _key):
        return MemoryObject.memories[self.memoryCode][_key]
    
    def getImage(self, _act, _pre_state, visited, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _pre_state, visited, path_trace)
        else:
            return MemoryObject.memories[self.memoryCode]

    def isAtomic(self):
        return self.teamMemory is None

    def mutate(self, mutateParams, parentTeam, teams, pMemAtom, learner_id):
        if any(item is None for item in (mutateParams, parentTeam, teams, pMemAtom, learner_id)):
            self.memoryCode=random.randint(0,len(MemoryObject.memories)-1)
            self.teamMemory=None
            print('0 valid_learners')
            return self

        if flip(pMemAtom):
            if self.memoryCode is not None:
                options = list(filter(lambda code: code != self.memoryCode, range(len(MemoryObject.memories))))
            else:
                options = len(MemoryObject.memories)
            if not self.isAtomic():
                self.teamMemory.inLearners.remove(str(learner_id))
            
            self.memoryCode = random.choice(options)
            self.teamMemory = None
        else:
            selection_pool = [t for t in teams if t is not self.teamMemory and t is not parentTeam]
            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamMemory.inLearners.remove(str(learner_id))
                
                self.teamMemory = random.choice(selection_pool)
                self.teamMemory.inLearners.append(str(learner_id))

        return self