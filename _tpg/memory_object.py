from math import tanh
import pickle
from uuid import uuid4
import numpy as np
import random
from _tpg.utils import flip, breakpoint, sigmoid

class _Fragment:
    """ state memory fragment """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls)

    def __init__(self, _key=np.array([0]), _state=np.array([[0.]]), _reward=0.):
        state = np.array(_state)
        key = np.array(_key)
        self.reward = _reward
        self.index = key
        self.fragment = state[key]
        self.id = str(uuid4())
    
    def __getitem__(self, key):
        return self.fragment[self.index==key]

    def keys(self):
        return self.index

    def values(self):
        return self.fragment
    
    # ここら辺をもっとRewardに沿った形で変更。
    def update(self, key, value):
        if isinstance(value, list): value = np.array(value)
        assert isinstance(value, np.ndarray)
        assert value.size==self.index.size

        self.fragment[[i for i,x in enumerate(self.index) if x in key]] = value

    def memorize(self, state, _reward):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        assert len(np.shape(state))==1, f'should {np.shape(state)} flatten'

        reward_unexpectancy = (self.reward-_reward)
        self.reward     -= reward_unexpectancy
        unexpectancy = abs(tanh(reward_unexpectancy))
        key = self.index[ self.index < state.size ]
        val = self.fragment[ self.index < state.size ]
        dif = val - state[ key ]
        diff = np.array(self.fragment)
        diff[[i for i,x in enumerate(self.index) if x in key]] = dif
        self.fragment   = self.fragment - diff*unexpectancy
        return diff, unexpectancy

    def recall(self, state):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        key = self.index[self.index<state.size]
        val = self.fragment[self.index<state.size]
        state[key] = val
        return state

class Fragment1(_Fragment):
    """actions fragment"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, _actionSequence:list=[]):
        assert isinstance(_actionSequence, list) or isinstance(_actionSequence, np.ndarray), f'{_actionSequence} is not list'

        self.fragment = np.array(_actionSequence)
        self.id = str(uuid4())
    
    def __getitem__(self, key):
        return self.fragment[key]

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return [ self.fragment[i] for i, item in enumerate(__o)]
        
    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return [ self.fragment[i] for i, item in enumerate(__o)]
        
    def keys(self):
        return range(len(self.fragment))

    def update(self, key, value):
        if isinstance(value, list): value = np.array(value)

        assert isinstance(value, np.ndarray)
        assert value.size==self.fragment.size

        self.fragment[key] = value

    @property
    def signal(self):
        return self.fragment

class Fragment2(_Fragment):
    """add compare"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'

        return

    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return

    def compare(self, state, _reward):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        assert len(np.shape(state))==1, f'should {np.shape(state)} flatten'

        reward_unexpectancy = (self.reward-_reward)
        unexpectancy = abs(tanh(reward_unexpectancy))
        key = self.index[ self.index < state.size ]
        val = self.fragment[ self.index < state.size ]
        dif = val - state[ key ]
        diff = np.array(self.fragment)
        diff[[i for i,x in enumerate(self.index) if x in key]] = dif
        return diff, unexpectancy

    @property
    def state(self):
        _state = np.zeros(np.max(self.index))
        _state[self.index]=self.fragment
        return _state

class _Memory:
    """states memory"""
    Fragment = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls.Fragment = _Fragment
        return super().__new__(cls)

    def __init__(self):
        fragment = self.__class__.Fragment()
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict={fragment.id:1.}

    def __getitem__(self, key):
        assert key in self.memories.keys(), f'{key} not in {self.memories.keys()}'
        return self.memories[key] # flagment

    def __delattr__(self, __name: str) -> None:
        del self.memories[__name]
        del self.weights[__name]

    def __contains__(self, __o):
        return __o in self.memories.keys()

    def __repr__(self) -> str:
        result = '<'
        for code in self.memories.keys():
            result+=f'{code}, '
        return result+'>'
    
    def append(self, _key, _state, _reward=0.):
        _key= list(set(_key))
        memory = self.__class__.Fragment(_key, _state, _reward)
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
        p = 0.9999-self.popus(_ignore)
        p=sigmoid(p)
        return random.choices(self.codes(_ignore), p)[0]

class Memory1(_Memory):
    """actions memory"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment1
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, actions=2):
        self.memories:dict={} # uuid:flagment
        self.weights:dict={}
        for i in range(actions):
            fragment = self.__class__.Fragment([i])
            self.memories[fragment.id]=fragment
            self.weights[fragment.id]=1.

    def __getitem__(self, key):
        assert key in self.memories.keys(), f'{key} not in {self.memories.keys()}'
        return self.memories[key] # fragment

    def append(self, _sequence):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        fragment = self.__class__.Fragment(_sequence)
        self.memories[fragment.id]    = fragment
        self.weights[fragment.id]     = 1.
        return fragment.id

    def choices(self, k=1, _ignore:list=[], )->list:
        p = 0.9999-self.popus(_ignore)
        p=sigmoid(p)
        return random.choices(self.codes(_ignore), p, k=k)

class Memory2(_Memory):
    """deactivate memorize"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment2
        return super().__new__(cls, *args, **kwargs)

class _MemoryObject:
    memories=_Memory()
    Team = None
    _instance = None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2
            cls._instance = True
            cls.Team = Team2

        return super().__new__(cls)
    
    def __init__(self, state=None, reward=0.):

        if isinstance(state, self.__class__.Team):
            self.teamMemory = state
            self.memoryCode = None
            return
        elif isinstance(state, self.__class__):
            self.memoryCode = state.memoryCode
            self.teamMemory = state.teamMemory
            return
        elif isinstance(state, np.ndarray):
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            self.memoryCode = self.__class__.memories.append(key, state, reward)
            self.teamMemory = None
            return
        else:
            self.memoryCode = self.__class__.memories.choice()
            self.teamMemory = None
            return

    def __eq__(self, __o: object) -> bool:

        if not isinstance(__o, self.__class__):   return False
        
        # The other object's action code must be equal to ours
        if self.memoryCode != __o.memoryCode:   return False
        
        # The other object's team action must be equal to ours
        if self.teamMemory != __o.teamMemory:   return False
        return True

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    
    def __getitem__(self, _key):
        return self.__class__.memories[self.memoryCode][_key]
    
    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}'
            self.__class__.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()               # 忘却確立計上
            return self.memoryCode

    def isAtomic(self):
        return self.teamMemory is None

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pMemAtom=None, learner_id=None):
        if None in (mutateParams, parentTeam, teams, pMemAtom, learner_id):
            self.memoryCode=self.__class__.memories.choice([self.memoryCode])
            self.teamMemory=None
            return self

        if flip(pMemAtom):
            if self.memoryCode is not None:
                _ignore = self.memoryCode
            else:
                _ignore = None
            
            if not self.isAtomic():
                self.teamMemory.inLearners.remove(str(learner_id))
            
            self.memoryCode = self.__class__.memories.choice([_ignore])
            self.teamMemory = None
        else:
            selection_pool = [t for t in teams if t is not self.teamMemory and t is not parentTeam]
            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamMemory.inLearners.remove(str(learner_id))
                
                self.teamMemory = random.choice(selection_pool)
                self.teamMemory.inLearners.append(str(learner_id))

        return self
    
    def backup(self, fileName):
        pickle.dump(self.__class__.memories, open(f'log/{fileName}-mem.pickle', 'wb'))

    @classmethod
    def emulate(cls, fileName):
        _memories = pickle.load(open(fileName, 'rb'))
        cls.memories = _memories
        return cls.__init__()

class MemoryObject(_MemoryObject):
    memories=Memory2()
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_1
            cls._instance = True
            cls.Team = Team2_1

        return super().__new__(cls)

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return self.memory + __o.memory

    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return self.memory - __o.memory
    
    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}'
            self.__class__.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()               # 忘却確立計上
            return self

    @property
    def memory(self):
        return self.__class__.memories[self.memoryCode]
