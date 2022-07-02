import pickle
import numpy as np
import random
from _tpg.utils import flip

class MemoryObject:
    memories=[{}]
    weights=np.array([], dtype=float)
    
    def __init__(self, state=0, initParam=None):
        from _tpg.team import Team2

        if isinstance(state, Team2):
            self.teamMemory = state
            self.memoryCode = None
            return
        elif isinstance(state, MemoryObject):
            self.memoryCode = state.memoryCode
            self.teamMemory = state.teamMemory
            return
        elif isinstance(state, int):
            # if _state > len(MemoryObject._memorys)-1: raise IndexError
            state %= len(MemoryObject.memories)
            self.memoryCode = state
            self.teamMemory = None
            return
        elif isinstance(state, np.ndarray):
            key = np.random.choices(range(state.size), random.randint(1, state.size-1))
            memory=dict(zip(key, state[key]))
            MemoryObject.memories.append(memory)
            np.append(MemoryObject.weights, 1)
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
            MemoryObject.weights[self.memoryCode]*=0.9
            MemoryObject.weights*=1.1
            for key, val in MemoryObject.memories[self.memoryCode].items():
                _pre_state[key]=val

            return _pre_state

    def isAtomic(self):
        return self.teamMemory is None

    def mutate(self, mutateParams, parentTeam, teams, pMemAtom, learner_id):
        if any(item is None for item in (mutateParams, parentTeam, teams, pMemAtom, learner_id)):
            self.memoryCode=random.randint(0,len(MemoryObject.memories), )
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
    
    @classmethod
    def buckup(cls, fileName):
        pickle.dump(cls, open(f'log/{fileName}.pickle', 'wb'))