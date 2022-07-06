import pickle
from uuid import uuid4
import numpy as np
import random
from _tpg.utils import flip

class Fragment:
    def __init__(self, _key=np.array([0]), _state=np.array([[0.]])):
        self.flagment = np.array(_state[_key])
        self.index = np.array(_key)
        self.id = str(uuid4())
    
    def __getitem__(self, key):
        return self.flagment[self.index==key][0]

    def keys(self):
        return self.index

    def values(self):
        return self.flagment
    
    def update(self, value):
        if not isinstance(value, np.ndarray): raise Exception('the data type is different')
        if isinstance(value, list): value = np.array(value)
        if value.size != self.index.size: raise Exception('the length is different')
        self.flagment = value

class Memory:
    def __init__(self):
        fragment = Fragment()
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict={fragment.id:1.}
        self.referenced:dict={fragment.id:0}

    def __getitem__(self, key):
        return self.memories[key] # flagment

    def __delattr__(self, __name: str) -> None:
        del self.memories[__name]
        del self.weights[__name]
        del self.referenced[__name]
    
    def append(self, _key, _state):
        memory = Fragment(_key, _state)
        self.memories[memory.id] = memory
        self.weights[memory.id] = 1.
        return memory.id
    
    def size(self):
        return len(self.weights)
    
    def codes(self, _ignore:list=None):
        if not _ignore is None: return list(filter(lambda x: not x in _ignore, self.weights.keys()))
        return list(self.weights.keys())
    
    def popus(self, _ignore:list=None):
        if not _ignore is None:
            codes = self.codes(_ignore)
            return np.array([val for key, val in self.weights.items() if key in codes])
            
        return np.array(list(self.weights.values()))

    def updateWeights(self, rate=1.01):
        self.weights = {x: val*rate for x, val in self.weights.items()}

    def oblivion(self):
        for key in [k for k, v in self.referenced.items() if v<1]:
            self.__delattr__(key)


    def choice(self, _ignore=None)->list:
        p = 1-self.popus(_ignore)
        p = p[p>0]
        if len(p)==0: return random.choice(self.codes(_ignore))
        return random.choices(self.codes(_ignore), p[p>0])[0]

class MemoryObject:
    memories=Memory()
    
    def __init__(self, state=1, initParam=None):
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
            self.memoryCode = MemoryObject.memories.choice()
            self.teamMemory = None
            return
        elif isinstance(state, np.ndarray):
            key = np.random.choices(range(state.size), random.randint(1, state.size-1))
            self.memoryCode = MemoryObject.memories.append(key, state)
            self.teamMemory = None

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, MemoryObject) and not isinstance(__o, np.ndarray): False
        memory:Fragment = MemoryObject.memories[self.memoryCode]

        for i in memory:
            if __o[i]!=memory[i]: False

        return True

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __getitem__(self, _key):
        return MemoryObject.memories[self.memoryCode][_key]
    
    def getImage(self, _act, _state, _bid, visited, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, _bid, visited, path_trace)
        else:
            MemoryObject.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            MemoryObject.memories.updateWeights()                    # 忘却確立計上

            return self.memoryCode, _bid[0]

    def isAtomic(self):
        return self.teamMemory is None

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pMemAtom=None, learner_id=None):
        if None in (mutateParams, parentTeam, teams, pMemAtom, learner_id):
            self.memoryCode=MemoryObject.memories.choice()
            self.teamMemory=None
            print('0 valid_learners')
            return self

        if flip(pMemAtom):
            if self.memoryCode is not None:
                MemoryObject.memories.referenced[self.memoryCode]-=1
                _ignore = self.memoryCode
            else:
                _ignore = None
            
            if not self.isAtomic():
                self.teamMemory.inLearners.remove(str(learner_id))
            
            self.memoryCode = MemoryObject.memories.choice(_ignore)
            MemoryObject.memories.referenced[self.memoryCode]+=1
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
    def backup(cls, fileName):
        pickle.dump(cls, open(f'log/{fileName}.pickle', 'wb'))
