from math import tanh
import pickle
from uuid import uuid4
import numpy as np
import random
from _tpg.utils import flip, breakpoint



class Fragment:
    def __init__(self, _key=np.array([0]), _state=np.array([[0.]])):
        state = np.array(_state)
        key = np.array(_key)
        self.reward = 0.
        self.fragment = state[key]
        self.index = key
        self.id = str(uuid4())
    
    def __getitem__(self, key):
        return self.fragment[self.index==key]

    def keys(self):
        return self.index

    def values(self):
        return self.fragment
    
    # ここら辺をもっとRewardに沿った形で変更。
    def update(self, key, value):
        assert isinstance(value, list) or isinstance(value, np.ndarray)
        if isinstance(value, list): value = np.array(value)
        assert value.size==self.index.size
        self.fragment[[i for i,x in enumerate(self.index) if x in key]] = value

    def memorize(self, state, _reward):
        assert isinstance(state, np.ndarray), state
        reward_unexpectancy = (self.reward-_reward)
        self.reward     -= reward_unexpectancy
        unexpectancy = abs(tanh(reward_unexpectancy))
        key = self.index[self.index<state.size]
        val = self.fragment[self.index<state.size]
        dif = val - state[key]
        diff = np.array(self.fragment)
        diff[[i for i,x in enumerate(self.index) if x in key]] = dif
        self.fragment   = self.fragment - diff*unexpectancy
        return diff, unexpectancy


    def recall(self, state):
        assert isinstance(state, np.ndarray)
        key = self.index[self.index<state.size]
        val = self.fragment[self.index<state.size]
        state[key] = val
        return state
    

class Memory:
    def __init__(self):
        fragment = Fragment()
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict={fragment.id:1.}

    def __getitem__(self, key):
        return self.memories[key] # flagment

    def __delattr__(self, __name: str) -> None:
        del self.memories[__name]
        del self.weights[__name]
    
    def append(self, _key, _state):
        _key= list(set(_key))
        memory = Fragment(_key, _state)
        self.memories[memory.id]    = memory
        self.weights[memory.id]     = 1.
        return memory.id
    
    def size(self):
        return len(self.weights)
    
    def codes(self, _ignore:list=None):
        if not _ignore is None: 
            return np.array(list(filter(lambda x: not x in _ignore, self.weights.keys())))
        return np.array(list(self.weights.keys()))
    
    def popus(self, _ignore:list=None):
        if not _ignore is None:
            codes = self.codes(_ignore)
            return np.array([val for key, val in self.weights.items() if key in codes])
        return np.array(list(self.weights.values()))

    def updateWeights(self, rate=1.01):
        self.weights = {x: val*rate for x, val in self.weights.items()}

    def oblivion(self, ignore):
        deleat_key = []
        for k, v in self.memories.items():
            if not k in ignore: deleat_key.append(k)
        for key in deleat_key:
            self.__delattr__(key)


    def choice(self, _ignore:list=[])->list:
        p = 1-self.popus(_ignore)
        if len(p[p>0])==0: return random.choice(self.codes(_ignore))
        return random.choices(self.codes(_ignore)[p>0], p[p>0])[0]


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
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            self.memoryCode = MemoryObject.memories.append(key, state)
            self.teamMemory = None


    def __eq__(self, __o: object) -> bool:

        if not isinstance(__o, MemoryObject):   return False
        
        # The other object's action code must be equal to ours
        if self.memoryCode != __o.memoryCode:   return False
        
        # The other object's team action must be equal to ours
        if self.teamMemory != __o.teamMemory:   return False
        return True

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __getitem__(self, _key):
        return MemoryObject.memories[self.memoryCode][_key]
    
    def getImage(self, _act, _state, visited, actVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, actVars=actVars, path_trace=path_trace)
        else:
            MemoryObject.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            MemoryObject.memories.updateWeights()               # 忘却確立計上
            return self.memoryCode

    def isAtomic(self):
        return self.teamMemory is None

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pMemAtom=None, learner_id=None):
        if None in (mutateParams, parentTeam, teams, pMemAtom, learner_id):
            self.memoryCode=MemoryObject.memories.choice([self.memoryCode])
            self.teamMemory=None
            # print('0 valid_learners')
            return self

        if flip(pMemAtom):
            if self.memoryCode is not None:
                # try:
                #     MemoryObject.memories.referenced[self.memoryCode]-=1
                # except:
                #     print('memory is crashed')
                _ignore = self.memoryCode
            else:
                _ignore = None
            
            if not self.isAtomic():
                self.teamMemory.inLearners.remove(str(learner_id))
            
            self.memoryCode = MemoryObject.memories.choice([_ignore])
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
        pickle.dump(cls, open(f'log/{fileName}-mem.pickle', 'wb'))
