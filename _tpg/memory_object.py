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

class Fragment1(_Fragment): # action memory
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
        return self.__class__(list(self.fragment)+list(__o.fragment))
        
    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return self.__class__(self.fragment[:-__o.size])

    # def __mul__(self, __o):
    #     assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
    #     fragment = list(self.fragment)
    #     for i in __o.fragment:
    #         fragment*=i
    #     return self.__class__(fragment)

    # def __pow__(self, __o):
    #     assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
    #     fragment = list()
    #     return self.__class__()

    def __and__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        _size = self.fragment.size if self.fragment.size < __o.fragment.size else __o.fragment.size
        return self.__class__(np.where(np.isnan(self.fragment[:_size]), __o.fragment[:_size], self.fragment[:_size]))

    def __or__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        if self.fragment.size > __o.fragment.size:
            __o_fragment=list(__o.fragment)+list(self.fragment[__o.fragment.size:])
            self_fragment = self.fragment
        else:
            self_fragment = list(self.fragment)+list(__o.fragment[self.fragment.size:])
            __o_fragment = __o.fragment
        return self.__class__(np.where(np.isnan(self_fragment), __o_fragment, self_fragment))

    def __xor__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        if self.fragment.size > __o.fragment.size:
            __o_fragment=list(__o.fragment)+list(self.fragment[__o.fragment.size:])
            self_fragment = self.fragment
        else:
            self_fragment = list(self.fragment)+list(__o.fragment[self.fragment.size:])
            __o_fragment = __o.fragment
        return self.__class__(np.where(np.isnan(__o_fragment), self_fragment, __o_fragment))

    def __invert__(self):
        return self.__class__(self.fragment[::-1])

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

class Fragment2(_Fragment): # sence memory
    """add compare"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return self.__class__(list(self.fragment)+list(__o.fragment))
        
    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        return self.__class__(self.fragment[:-__o.size])

    def __mul__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        fragment = np.array(self.fragment)
        for i in __o.fragment:
            fragment*=i
        return self.__class__(fragment)

    def __truediv__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        fragment = np.array(self.fragment)
        fragment[:-__o.fragment.size]=np.nan
        return self.__class__(fragment)

    def __and__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        _size = self.fragment.size if self.fragment.size < __o.fragment.size else __o.fragment.size
        return self.__class__(np.where(np.isnan(self.fragment[:_size]), __o.fragment[:_size], self.fragment[:_size]))

    def __or__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        if self.fragment.size > __o.fragment.size:
            __o_fragment=list(__o.fragment)+list(self.fragment[__o.fragment.size:])
            self_fragment = self.fragment
        else:
            self_fragment = list(self.fragment)+list(__o.fragment[self.fragment.size:])
            __o_fragment = __o.fragment
        return self.__class__(np.where(np.isnan(self_fragment), __o_fragment, self_fragment))

    def __xor__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} must be {self.__class__}'
        if self.fragment.size > __o.fragment.size:
            __o_fragment=list(__o.fragment)+list(self.fragment[__o.fragment.size:])
            self_fragment = self.fragment
        else:
            self_fragment = list(self.fragment)+list(__o.fragment[self.fragment.size:])
            __o_fragment = __o.fragment
        return self.__class__(np.where(np.isnan(__o_fragment), self_fragment, __o_fragment))

    def __invert__(self):
        return self.__class__(self.fragment[::-1])

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

class Fragment3(_Fragment): # story memory

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, _actObj, _memObjs=[]):
        self.actObj = _actObj
        self.fragment = list(_memObjs)
        self.id = str(uuid4())
    
    def __getitem__(self, __key):
        return self.fragment[__key]

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
        for k in self.memories:
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
        fragment = self.__class__.Fragment([np.nan])
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict ={fragment.id:0.}
        self.nan = fragment.id
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

    @property
    def NaN(self):
        return self.memories[self.nan]

class Memory2(_Memory):
    """deactivate memorize"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment2
        return super().__new__(cls, *args, **kwargs)

class Memory3(_Memory):
    ActionObject=None
    MemoryObject=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls.Fragment = Fragment3
            cls.ActionObject = ActionObject2
            cls.MemoryObject = MemoryObject1
        return super().__new__(cls, *args, **kwargs)

    def append(self, _actionSequence, _memorySequence, _rewardSequence):
        # assert isinstance(_actObj, self.__class__.ActionObject) and isinstance(_memObjs[0], self.__class__.MemoryObject), \
        #     f'{_actObj} should {self.__class__.ActionObject} and {_memObjs} list of {self.__class__.MemoryObject}'
        actObj = self.__class__.ActionObject(_actionSequence)
        memObjs = [ self.__class__.MemoryObject(memory) for memory in _memorySequence ]
        emotion = sigmoid(sum(_rewardSequence)*0.01)

        fragment = self.__class__.Fragment(actObj,memObjs)
        self.memories[fragment.id]=fragment
        self.weights[fragment.id] =1.-emotion
        return fragment.id

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
    
    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}'
            self.__class__.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()               # 忘却確立計上
            return self

    def recall(self, state):
        return self.__class__.memories[self.memoryCode].recall(state)

    @property
    def memory(self):
        return self.__class__.memories[self.memoryCode]

    @property
    def reward(self):
        return self.__class__.memories[self.memoryCode].reward
    
    @property
    def state(self):
        return self.__class__.memories[self.memoryCode].state

class MemoryObject1(MemoryObject):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_2
            cls._instance = True
            cls.Team = Team2_2
            cls.memories = Memory2()
            # cls.hippocampus = Hippocampus()s

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

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.memory + __o.memory)
    
    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.memory - __o.memory)
    
    def __and__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.memory & __o.memory)
    
    def __or__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.memory | __o.memory)
    
    def __xor__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.memory ^ __o.memory)
    
    def __invert__(self):
        # assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(~self.memory)
    
    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}'
            self.__class__.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()               # 忘却確立計上
            return self

class _ActionObject:
    """
    Action  Object has a program to produce a value for the action, program doesn't
    run if just a discrete action code.
    """
    actions=[0]
    Team = None
    _instance = None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import _Team
            cls._instance = True
            cls.Team = _Team
        return super().__new__(cls)
    '''
    An action object can be initalized by:
        - Copying another action object
        - Passing an index into the action codes in initParams as the action
        - Passing a team as the action
    '''
    def __init__(self,
        initParams:dict or int =None,
        action = None,
        _task='task'
    ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(action, self.__class__.Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return
        # An int means the action is an index into the action codes in initParams
        else:
            try:
                self.actionCode=random.choice(self.__class__.actions)
                self.teamAction=None
            except:
                print('諦めな・・・')
    '''
    An ActionObject is equal to another object if that object:
        - is an instance of the ActionObject class
        - has the same action code
        - has the same team action
    '''
    def __eq__(self, __o:object)->bool:

        # The other object must be an instance of the ActionObject class
        if not isinstance(__o, self.__class__):    return False
        
        # The other object's action code must be equal to ours
        if self.actionCode != __o.actionCode:     return False
        
        # The other object's team action must be equal to ours
        if self.teamAction != __o.teamAction:     return False

        return True

    '''
    Negate __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def __str__(self):
        return "TeamAction {} ActionCode: {}".format(
            self.teamAction if self.teamAction is not None else 'None',
            self.actionCode if self.actionCode is not None else 'None'
        )

    def zeroRegisters(self):
        try:
            self.registers = np.zeros(len(self.registers), dtype=float)
        except:
            pass

    def getAction(self, state, visited, actVars=None, path_trace=None):
        """
        Returns the action code, and if applicable corresponding real action(s).
        """
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    def isAtomic(self):
        """
        Returns true if the action is atomic, otherwise the action is a team.
        """
        return self.teamAction is None


    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        """
        Change action to team or atomic action.
        """
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(self.__class__.actions)
            self.teamAction = None
            print('0 valid_learners')

            return self

        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                options = list(filter(lambda code: code != self.actionCode,self.__class__.actions))
            else:
                options = self.__class__.actions

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = random.choice(options)
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            # If we have a valid set of options choose from them
            if len(selection_pool) > 0:
                # let our current team know we won't be pointing to them anymore
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

class ActionObject1(_ActionObject):
    actions=Memory1()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_1
            cls._instance = True
            cls.Team = Team1_1

        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
        action = None,
        _task='task',
        initParams:dict or int =None
    ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(action, self.__class__.Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return
        # An int means the action is an index into the action codes in initParams
        elif isinstance(action, str):
            assert action in self.__class__.actions, f'{action} not in {self.__class__.actions}'
            self.actionCode = action
            self.teamAction = None
            return
        else:
            try:
                self.actionCode=self.__class__.actions.choice()
                self.teamAction=None
            except:
                print('諦めな・・・')

    def __getitem__(self, _key):
        return self.__class__.actions[self.actionCode][_key]
    
    def getAction(self, _state, visited, actVars, path_trace=None):
        if self.teamAction is not None:
            return self.teamAction.act(_state, visited, actVars=actVars, path_trace=path_trace)
        else:
            assert self.actionCode in self.__class__.actions, f'{self.actionCode} is not in {self.__class__.actions}'
            self.__class__.actions.weights[self.actionCode]*=0.9 # 忘却確立減算
            self.__class__.actions.updateWeights()               # 忘却確立計上
            return self

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = self.__class__.actions.choice([self.actionCode])
            self.teamAction = None
            return self

        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                _ignore = self.actionCode
            else:
                _ignore = None

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = self.__class__.actions.choice([_ignore])
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                self.teamAction.inLearners.append(str(learner_id))
       
        return self
 
    def backup(self, fileName):
        pickle.dump(self.__class__.actions, open(f'log/{fileName}-act.pickle', 'wb'))

    @classmethod
    def emulate(cls, fileName):
        _actions = pickle.load(open(fileName, 'rb'))
        cls.actions = _actions
        return cls.__init__()

    @property
    def action(self):
        return self.__class__.actions[self.actionCode]

class ActionObject2(ActionObject1):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_2
            cls._instance = True
            cls.Team = Team1_2
            cls.actions=Memory1()

        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
        action = None,
        _task='task',
        initParams:dict or int =None
    ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(action, self.__class__.Team):
            self.teamAction = action
            self.actionCode = None
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return
        # An int means the action is an index into the action codes in initParams
        elif isinstance(action, str):
            assert action in self.__class__.actions, f'{action} not in {self.__class__.actions}'
            self.actionCode = action
            self.teamAction = None
            return
        elif isinstance(action, list):
            self.actionCode = self.__class__.actions.append(action)
            self.teamAction = None
            return
        else:
            try:
                self.actionCode=self.__class__.actions.choice()
                self.teamAction=None
            except:
                print('諦めな・・・')

    def __add__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.action + __o.action)
    
    def __sub__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.action - __o.action)
    
    def __and__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.action & __o.action)
    
    def __or__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.action | __o.action)
    
    def __xor__(self, __o):
        assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(self.action ^ __o.action)
    
    def __invert__(self):
        # assert isinstance(__o, self.__class__), f'{__o} is not {self.__class__}'
        return self.__class__(~self.action)
    
    @property
    def NaN(self):
        return self.__class__(self.__class__.actions.nan)

class Hippocampus:
    _instance = None
    Memory = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Memory = Memory3
        return super().__new__(cls)

    def __init__(self) -> None:
        self.real = self.__class__.Memory()
        self.mind = self.__class__.Memory()

    def __call__(self):
        pass