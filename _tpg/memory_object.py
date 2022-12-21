from math import tanh
import pickle
from uuid import uuid4, UUID
import numpy as np
import random
from _tpg.utils import flip, breakpoint, sigmoid, abstract, _Logger
import logging

# GAMMA = ['+','-','*','**','/','//','%','<','<=','<<','>','>=','>>','&','|','^','~','np.nan']
GAMMA =   [ 0 , 1 , 2 , 3  , 4 , 5  , 6 , 7 , 8  , 9  , 10, 11 , 12 , 13, 14, 15, 16, 17]



class _Fragment(_Logger):
    """ state memory fragment """
    _instance = None
    _logger = None

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

    def __init__(self, _actionSequence:list=[np.nan]):
        assert isinstance(_actionSequence, list) or isinstance(_actionSequence, np.ndarray), f'{_actionSequence} is not list'

        self.fragment = np.array(_actionSequence)
        self.id = str(uuid4())
    
    def __getitem__(self, key):
        return self.fragment[key]

    def values(self):
        return self.fragment

    def keys(self):
        return range(self.fragment.size)

    def update(self, key, value):
        if isinstance(value, list): value = np.array(value)

        assert isinstance(value, np.ndarray)
        assert value.size==self.fragment.size

        self.fragment[key] = value

class Fragment1_1(Fragment1):
    """operation implement"""
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, _sequence:list=[], _reward=np.nan, _weight=1.):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        sequence = [_reward]+list(_sequence)
        self.fragment = np.array(sequence)
        self.weight = _weight
        self._id = uuid4()

    def __add__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                return self.__class__(np.where(np.isnan(_o), self.fragment, _o))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                return self.__class__(np.where(np.isnan(__o.fragment), _self, __o.fragment))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(_o), self.fragment, _o))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(__o), _self, __o))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), self.fragment, __o))
        else:
            return NotImplemented
    
    def __radd__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(self.fragment), _o, self.fragment))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(_self), __o, _self))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), __o, self.fragment))
        else:
            return NotImplemented   
    
    def __sub__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                return self.__class__(np.where(np.isnan(_o), self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                return self.__class__(np.where(np.isnan(__o.fragment), _self, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(_o), self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(__o), _self, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), self.fragment, np.nan))
        else:
            return NotImplemented
    
    def __rsub__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(self.fragment), _o, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(_self), __o, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), __o, np.nan))
        else:
            return NotImplemented
    
    def __mul__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = []
            for _o in __o.fragment:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(self.fragment)*int(_o)
            return self.__class__(fragment)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _o in __o:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(self.fragment)*int(_o)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(list(self.fragment) * int(__o))
        else:
            return NotImplemented
    
    def __rmul__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(__o)*int(_self)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(__o)*int(_self)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __pow__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            for _o in __o.fragment:
                if _o is not np.nan:
                    fragment*=int(_o)
            return self.__class__(fragment)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            for _o in __o:
                if _o is not np.nan:
                    fragment*=int(_o)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment)
            return self.__class__(list(self.fragment) * int(__o))
        else:
            return NotImplemented
    
    def __rpow__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    fragment*=int(_self)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    fragment*=int(_self)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __truediv__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = []
            for _o in __o.fragment:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_o) == 0: continue
                    fragment+=list(self.fragment)[:int(self.size/_o)]
            return self.__class__(fragment)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _o in __o:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_o) == 0: continue
                    fragment+=list(self.fragment)[:int(self.size/_o)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment)
            return self.__class__(list(self.fragment)[:int(self.size/__o)])
        else:
            return NotImplemented
    
    def __rtruediv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_self) == 0: continue
                    fragment+=list(__o)[:int(len(__o)/_self)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0: return self.__class__([])
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_self) == 0: continue
                    fragment+=list(__o)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __floordiv__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            for _o in __o.fragment:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[:int(len(fragment)/_o)]
            return self.__class__(fragment)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            for _o in __o:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[:int(len(fragment)/_o)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment)
            return self.__class__(list(self.fragment)[:int(self.size/__o)])
        else:
            return NotImplemented
    
    def __rfloordiv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[:int(len(fragment)/_self)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__([])
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[:int(len(fragment)/_self)]
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __mod__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            for _o in __o.fragment:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[-int(len(fragment)%_o):]
            return self.__class__(fragment)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            for _o in __o:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[-int(len(fragment)%_o):]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment)
            return self.__class__(list(self.fragment)[-int(self.size%__o):])
        else:
            return NotImplemented
    
    def __rmod__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[-int(len(fragment)%_self):]
            return self.__class__(fragment)

        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__([])
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[-int(len(fragment)%_self):]
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __lt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size < __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size < len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size < __o
        else:
            return NotImplemented
    
    def __rlt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) < self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o < self.size
        else:
            return NotImplemented
    
    def __le__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size <= __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size <= len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size <= __o
        else:
            return NotImplemented
    
    def __rle__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) <= self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o <= self.size
        else:
            return NotImplemented

    def __lshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(list(self.fragment) + list(__o.fragment))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(self.fragment)+list(__o))
        elif isinstance(__o, int):
            return self.__class__(list(self.fragment) + [np.nan]*__o)
        elif isinstance(__o, float):
            return self.__class__(list(self.fragment) + [np.nan]*int(__o))
        else:
            return NotImplemented
    
    def __rlshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(__o) + list(self.fragment))
        elif isinstance(__o, int):
            return self.__class__([np.nan]*__o + list(self.fragment))
        elif isinstance(__o, float):
            return self.__class__([np.nan]*int(__o) + list(self.fragment))
        else:
            return NotImplemented
    
    def __gt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size > __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size > len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size > __o
        else:
            return NotImplemented
    
    def __rgt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) > self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o > self.size
        else:
            return NotImplemented
    
    def __ge__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size >= __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size >= len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size >= __o
        else:
            return NotImplemented
    
    def __rge__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) >= self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o >= self.size
        else:
            return NotImplemented
    
    def __rshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(list(__o.fragment) + list(self.fragment))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(__o)+list(self.fragment))
        elif isinstance(__o, int):
            return self.__class__([np.nan]*__o+list(self.fragment))
        elif isinstance(__o, float):
            return self.__class__([np.nan]*int(__o)+list(self.fragment))
        else:
            return NotImplemented
    
    def __rrshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(self.fragment)+list(__o))
        elif isinstance(__o, int):
            return self.__class__(list(self.fragment)+[np.nan]*__o)
        elif isinstance(__o, float):
            return self.__class__(list(self.fragment)+[np.nan]*int(__o))
        else:
            return NotImplemented

    def __and__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _self = self.fragment[:__o.size]
                _is = (~np.isnan(_self))&(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _self = self.fragment[:len(__o)]
                _is = (~np.isnan(_self))&(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __rand__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _self = self.fragment[:len(__o)]
                _is = (~np.isnan(_self))&(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented
    
    def __or__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __ror__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, _o, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o))
                return self.__class__(np.where(_is, __o, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __xor__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), np.nan, self.fragment))
        else:
            return NotImplemented

    def __rxor__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, _o, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o))
                return self.__class__(np.where(_is, __o, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), np.nan, self.fragment))
        else:
            return NotImplemented

    def __invert__(self):
        return self.__class__(self.fragment[::-1])

    @property
    def size(self):
        return self.fragment.size

    @property
    def id(self):
        return str(self._id)

class Fragment2(_Fragment): # sence memory
    """add compare"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, _key=[0], _state=np.array([[0.]]), _reward=0.):
        state = np.array(_state)
        if _key is None: _key = range(state.size)
        key = np.array(_key)
        self.reward = _reward
        self.index = key
        self.fragment = state[key]
        self.id = str(uuid4())

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

class Fragment2_1(Fragment2):
    """operation implement"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, _state=[], _reward=0., _key=None):
        state = np.array(_state)
        if _key is None: _key = range(state.size)
        key = np.array(list(set(_key)))
        self.fragment = np.full(state.size, np.nan)
        self.fragment[key] = state[key]
        self.reward = _reward
        self.id = str(uuid4())
    
    def __add__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                return self.__class__(np.where(np.isnan(_o), self.fragment, _o), self.reward+__o.reward)
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                return self.__class__(np.where(np.isnan(__o.fragment), _self, __o.fragment), self.reward+__o.reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(_o), self.fragment, _o), self.reward)
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(__o), _self, __o), self.reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), self.fragment, __o), self.reward)
        else:
            return NotImplemented
    
    def __radd__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(self.fragment), _o, self.fragment), self.reward)
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(_self), __o, _self), self.reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), __o, self.fragment), self.reward)
        else:
            return NotImplemented   
    
    def __sub__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                return self.__class__(np.where(np.isnan(_o), self.fragment, np.nan), self.reward-__o.reward)
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                return self.__class__(np.where(np.isnan(__o.fragment), _self, np.nan), self.reward-__o.reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(_o), self.fragment, np.nan), self.reward)
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(__o), _self, np.nan), self.reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), self.fragment, np.nan), self.reward)
        else:
            return NotImplemented
    
    def __rsub__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                return self.__class__(np.where(np.isnan(self.fragment), _o, np.nan), -self.reward)
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                return self.__class__(np.where(np.isnan(_self), __o, np.nan), -self.reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isnan(self.fragment), __o, np.nan), -self.reward)
        else:
            return NotImplemented
    
    def __mul__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = []
            reward = 0.
            for _o in __o.fragment:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(self.fragment)*int(_o)
                    reward +=self.reward*_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            reward = 0.
            for _o in __o:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(self.fragment)*int(_o)
                    reward +=self.reward*_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(list(self.fragment) * int(__o), self.reward*__o)
        else:
            return NotImplemented
    
    def __rmul__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(__o)*int(_self)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    fragment+=list(__o)*int(_self)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __pow__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o.fragment:
                if _o is not np.nan:
                    fragment*=int(_o)
                    reward*=_o
            return self.__class__(fragment,reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o:
                if _o is not np.nan:
                    fragment*=int(_o)
                    reward *= _o
            return self.__class__(fragment, reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment, self.reward)
            return self.__class__(list(self.fragment) * int(__o), self.reward*__o)
        else:
            return NotImplemented
    
    def __rpow__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    fragment*=int(_self)
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    fragment*=int(_self)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __truediv__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = []
            reward = 0.
            for _o in __o.fragment:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_o) == 0: continue
                    fragment+=list(self.fragment)[:int(self.size/_o)]
                    reward += self.reward/_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            reward = 0.
            for _o in __o:
                if _o is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_o) == 0: continue
                    fragment+=list(self.fragment)[:int(self.size/_o)]
                    reward += self.reward/_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment, self.reward)
            return self.__class__(list(self.fragment)[:int(self.size/__o)], self.reward/__o)
        else:
            return NotImplemented
    
    def __rtruediv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_self) == 0: continue
                    fragment+=list(__o)[:int(len(__o)/_self)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0: return self.__class__([])
            fragment = []
            for _self in self.fragment:
                if _self is np.nan:
                    fragment+=[np.nan]
                else:
                    if int(_self) == 0: continue
                    fragment+=list(__o)
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __floordiv__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o.fragment:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[:int(len(fragment)/_o)]
                    reward/=_o
            return self.__class__(fragment,reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[:int(len(fragment)/_o)]
                    reward/=_o
            return self.__class__(fragment,reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment, self.reward)
            return self.__class__(list(self.fragment)[:int(self.size/__o)], self.reward/__o)
        else:
            return NotImplemented
    
    def __rfloordiv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[:int(len(fragment)/_self)]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__([])
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[:int(len(fragment)/_self)]
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __mod__(self, __o):
        if isinstance(__o, self.__class__):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o.fragment:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[-int(len(fragment)%_o):]
                    reward%=_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(self.fragment)
            reward = self.reward
            for _o in __o:
                if _o is not np.nan:
                    if int(_o) == 0: continue
                    fragment=fragment[-int(len(fragment)%_o):]
                    reward%=_o
            return self.__class__(fragment, reward)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__(self.fragment, self.reward)
            return self.__class__(list(self.fragment)[-int(self.size%__o):], self.reward%__o)
        else:
            return NotImplemented
    
    def __rmod__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[-int(len(fragment)%_self):]
            return self.__class__(fragment)
        elif isinstance(__o, int) or isinstance(__o, float):
            if int(__o) == 0 or __o is np.nan: return self.__class__([])
            fragment = list(__o)
            for _self in self.fragment:
                if _self is not np.nan:
                    if int(_self) == 0: continue
                    fragment=fragment[-int(len(fragment)%_self):]
            return self.__class__(fragment)
        else:
            return NotImplemented
    
    def __lt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size < __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size < len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size < __o
        else:
            return NotImplemented
    
    def __rlt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) < self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o < self.size
        else:
            return NotImplemented
    
    def __le__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size <= __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size <= len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size <= __o
        else:
            return NotImplemented
    
    def __rle__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) <= self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o <= self.size
        else:
            return NotImplemented

    def __lshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(list(self.fragment) + list(__o.fragment), self.reward+__o.reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(self.fragment)+list(__o), self.reward)
        elif isinstance(__o, int):
            return self.__class__(list(self.fragment) + [np.nan]*__o, self.reward)
        elif isinstance(__o, float):
            return self.__class__(list(self.fragment) + [np.nan]*int(__o), self.reward)
        else:
            return NotImplemented
    
    def __rlshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(__o) + list(self.fragment))
        elif isinstance(__o, int):
            return self.__class__([np.nan]*__o + list(self.fragment))
        elif isinstance(__o, float):
            return self.__class__([np.nan]*int(__o) + list(self.fragment))
        else:
            return NotImplemented
    
    def __gt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size > __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size > len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size > __o
        else:
            return NotImplemented
    
    def __rgt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) > self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o > self.size
        else:
            return NotImplemented
    
    def __ge__(self, __o):
        if isinstance(__o, self.__class__):
            return self.size >= __o.size
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.size >= len(__o)
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.size >= __o
        else:
            return NotImplemented
    
    def __rge__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return len(__o) >= self.size
        elif isinstance(__o, int) or isinstance(__o, float):
            return __o >= self.size
        else:
            return NotImplemented
    
    def __rshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(list(__o.fragment) + list(self.fragment), __o.reward+self.reward)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(__o)+list(self.fragment), self.reward)
        elif isinstance(__o, int):
            return self.__class__([np.nan]*__o+list(self.fragment), self.reward)
        elif isinstance(__o, float):
            return self.__class__([np.nan]*int(__o)+list(self.fragment), self.reward)
        else:
            return NotImplemented
    
    def __rrshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            return self.__class__(list(self.fragment)+list(__o))
        elif isinstance(__o, int):
            return self.__class__(list(self.fragment)+[np.nan]*__o)
        elif isinstance(__o, float):
            return self.__class__(list(self.fragment)+[np.nan]*int(__o))
        else:
            return NotImplemented

    def __and__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _self = self.fragment[:__o.size]
                _is = (~np.isnan(_self))&(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _self = self.fragment[:len(__o)]
                _is = (~np.isnan(_self))&(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __rand__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _self = self.fragment[:len(__o)]
                _is = (~np.isnan(_self))&(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
            else:
                _is = (~np.isnan(self.fragment))&(~np.isnan(__o[:self.size]))
                return self.__class__(np.where(_is, self.fragment, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented
    
    def __or__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __ror__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))|(~np.isnan(_o))
                return self.__class__(np.where(_is, _o, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))|(~np.isnan(__o))
                return self.__class__(np.where(_is, __o, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), self.fragment, np.nan))
        else:
            return NotImplemented

    def __xor__(self, __o):
        if isinstance(__o, self.__class__):
            if self > __o:
                _o = list(__o.fragment)+list(self.fragment[__o.size:])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o.fragment[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o.fragment))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, self.fragment, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o))
                return self.__class__(np.where(_is, _self, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), np.nan, self.fragment))
        else:
            return NotImplemented

    def __rxor__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list):
            if self > __o:
                _o = list(__o)+list(self.fragment[len(__o):])
                _is = (~np.isnan(self.fragment))^(~np.isnan(_o))
                return self.__class__(np.where(_is, _o, np.nan))
            else:
                _self = list(self.fragment)+list(__o[self.size:])
                _is = (~np.isnan(_self))^(~np.isnan(__o))
                return self.__class__(np.where(_is, __o, np.nan))
        elif isinstance(__o, int) or isinstance(__o, float):
            return self.__class__(np.where(np.isclose(self.fragment, [__o]*self.size), np.nan, self.fragment))
        else:
            return NotImplemented

    def __invert__(self):
        return self.__class__(self.fragment[::-1], -self.reward)
    
    def __getitem__(self, key):
        return self.fragment[key]

    def keys(self):
        return np.argwhere(~np.isnan(self.fragment)).flatten()

    def values(self):
        return self.fragment[~np.isnan(self.fragment)]

    def update(self, key, value):
        if isinstance(value, list): value = np.array(value)
        assert isinstance(value, np.ndarray)
        assert len(value)==len(key)
        self.fragment[key] = value

    def memorize(self, state, _reward):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        assert len(np.shape(state))==1, f'should {np.shape(state)} flatten'

        reward_unexpectancy = (self.reward-_reward)
        self.reward -= reward_unexpectancy
        unexpectancy = abs(tanh(reward_unexpectancy))
        _this = self | state
        diff = _this.fragment-state
        self.fragment = self.fragment - diff*unexpectancy
        return diff, unexpectancy

    def compare(self, state, _reward):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        assert len(np.shape(state))==1, f'should {np.shape(state)} flatten'

        reward_unexpectancy = (self.reward-_reward)
        unexpectancy = abs(tanh(reward_unexpectancy))
        dif = np.power(self.fragment-state, 2)
        diff = dif[dif<np.inf].sum()
        
        return diff, unexpectancy

    def recall(self, state):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        return state + self

    @property
    def size(self):
        return self.fragment.size

    @property
    def state(self):
        return self.fragment

class Fragment3(Fragment1_1):
    """story memory"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, _sequence:list or np.ndarray=[], _head=np.nan, _weight=1.):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        sequence = [_head]+list(_sequence)
        self.fragment = np.array(sequence)
        self.weight = _weight
        self._id = uuid4()

    def __hash__(self) -> int:
        return int(self._id)

    def compare(self, state):
        assert isinstance(state, np.ndarray), f'should be ndarray or list {state}'
        assert len(np.shape(state))==1, f'should {np.shape(state)} flatten'

        dif = np.power(self.fragment-state, 2)
        diff = dif[dif<np.inf].sum()
        
        return diff

    def recall(self, state):
        assert isinstance(state, np.ndarray), f'should be ndarray {state}'
        return state + self

    @property
    def id(self):
        return str(self._id)

class Fragment3_1(Fragment3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=True
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, _sequence:list or np.ndarray=[], _weight=1.):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        # sequence = [_head]+list(_sequence)
        self.fragment = np.array(_sequence)
        self.weight = _weight
        self._id = uuid4()

class _Memory(_Logger):
    """states memory"""
    Fragment = None
    # _instance = None
    # _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls.Fragment = _Fragment
        return super().__new__(cls, *args, **kwargs)

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
        for code in self.memories:
            result+=f'{code}, '
        return result+'>'
    
    def __str__(self) -> str:
        result = '<'
        for code in self.memories.values():
            result+=f'{code.fragment}, '
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
        # self.debug(f'oblivion:{str(self)}')
        # breakpoint('oblivion')
        for k in self.memories:
            if not k in ignore: deleat_key.append(k)
        for key in deleat_key:
            self.__delattr__(key)

    def choice(self, _ignore:list=[])->list:        
        p = 1-self.popus(_ignore)
        p=sigmoid(p)+0.0001
        return random.choices(self.codes(_ignore), p)[0]

class Memory1(_Memory):
    """actions memory"""
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment1
            # cls.Fragment._logger = cls._logger
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

class Memory1_1(Memory1):
    """operation implement"""
    # _instance=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment1_1
            # cls.Fragment._logger = cls._logger

        return super().__new__(cls, *args, **kwargs)

    def values(self):
        return self.memories.values()

class Memory1_2(Memory1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment1_1
            # cls.Fragment._logger = cls._logger

        return super().__new__(cls, *args, **kwargs)
    

    def __init__(self, _range=1, actions=None):
        fragment = self.__class__.Fragment()
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict ={fragment.id:0.}
        self.nan = fragment.id
        for i in range(_range):
            fragment = self.__class__.Fragment([i])
            self.memories[fragment.id]=fragment
            self.weights[fragment.id]=1.

class Memory2(_Memory):
    """deactivate memorize"""
    # _instance=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment2
            # cls.Fragment._logger = cls._logger
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        fragment = self.__class__.Fragment()
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict ={fragment.id:0.}

class Memory2_1(Memory2):
    """opelation implement"""
    # _instance=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment2_1
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        fragment = self.__class__.Fragment(_state=[np.nan])
        self.memories:dict={fragment.id:fragment} # uuid:flagment
        self.weights:dict ={fragment.id:0.}
        self.nan = fragment.id
        
    def append(self, _state, _reward=0., _key=None):
        memory = self.__class__.Fragment(_state, _reward, _key)
        self.memories[memory.id]    = memory
        self.weights[memory.id]     = 1.
        return memory.id

    @property
    def NaN(self):
        return self.memories[self.nan]

class Memory3(Memory1_2):
    """ Fragment object management

        Attribute:
            memories: fragments
            nan: Fragment([nan])
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment3
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, primitive=[]):
        assert isinstance(primitive, list), f'primitive is not list'
        fragment = self.__class__.Fragment(_weight=0.)
        self.memories:dict={fragment._id:fragment} # uuid:flagment
        self.nan = fragment
        for pri in primitive:
            fragment = self.__class__.Fragment([pri], _weight=0.)
            self.memories[fragment._id]=fragment

    def __contains__(self, __o):
        """
            property:
                __o: uuid
        """
        return __o in list(self.memories.keys())

    def __delattr__(self, __name: str) -> None:
        del self.memories[__name]

    def __len__(self):
        return len(self.memories)

    def append(self, _sequence, _head=np.nan, _weight=1.):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        fragment = self.__class__.Fragment(_sequence, _head=_head, _weight=_weight)
        self.memories[fragment._id] = fragment
        return fragment._id

    def codes(self, _ignore:list=None):
        if _ignore : 
            return np.array(list(filter(lambda x: not x in _ignore, self.keys())))
        return np.array(list(self.keys()))
    
    def popus(self, _ignore:list=None):
        """ return weight with ignore proccessing """
        if _ignore:
            # codes = self.codes(_ignore)
            return np.array([frg.weight for frg in self.values() if frg._id not in _ignore])
        return np.array(self.weights)

    def updateWeights(self, rate=1.1):
        # self.weights = {x: val*rate for x, val in self.weights.items()}
        for fragment in self.values():
            fragment.weight*=rate

    def choices(self, k=1, _ignore:list=None)->list:
        p = 1-self.popus(_ignore)
        p=sigmoid(p)+0.0001
        return random.choices(self.codes(_ignore), p, k=k)

    def keys(self):
        return self.memories.keys()

    def values(self):
        return self.memories.values()

    @property
    def size(self):
        return len(self.memories)

    @property
    def weights(self):
        return [fragment.weight for fragment in self.values()]

class Memory3_1(Memory3):
    """ Fragment object management

        Attribute:
            memories: fragments
            nan: Fragment([nan])
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment3_1
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, primitive=[]):
        assert isinstance(primitive, list), f'primitive is not list'
        fragment = self.__class__.Fragment(_weight=0.)
        self.memories:dict={fragment._id:fragment} # uuid:flagment
        self.nan = fragment
        for pri in primitive:
            fragment = self.__class__.Fragment([pri], _weight=0.)
            self.memories[fragment._id]=fragment

    def append(self, _sequence, _weight=1.):
        assert isinstance(_sequence, list) or isinstance(_sequence, np.ndarray), f'{_sequence} is not list'
        fragment = self.__class__.Fragment(_sequence, _weight=_weight)
        self.memories[fragment._id] = fragment
        return fragment._id

class Memory3_2(Memory3_1):
    """ Fragment object management

        Attribute:
            memories: fragments
            nan: Fragment([nan])
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            cls.Fragment = Fragment3_1
        return super().__new__(cls, *args, **kwargs)

    def choices(self, k=1, _ignore:list=None)->list:
        p = 1-self.popus(_ignore)
        p=sigmoid(p)+0.0001
        return random.choices(self.codes(_ignore), p=p, k=k)
        # return random.choices(self.codes(_ignore), k=k)

    def updateWeights(self, rate=1.01):
        # self.weights = {x: val*rate for x, val in self.weights.items()}
        for fragment in self.values():
            fragment.weight*=rate

class _MemoryObject(_Logger):
    Team = None
    Memory = None
    memories=None
    # _instance = None
    # _logger = None
    _nan=None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2
            cls._instance = True
            cls.Team = Team2
            cls.Memory = _Memory
            cls.memories = cls.Memory()
            cls._nan = cls.memories.nan

        return super().__new__(cls, *args, **kwargs)
    
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

    def __str__(self):
        return f"TeamMemory {self.teamMemory} MemoryCode: {self.memoryCode}"

    def __getitem__(self, _key):
        return self.__class__.memories[self.memoryCode][_key]
    
    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}, {self.__class__}'
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

    @classmethod
    @property
    def NaN(cls):
        return cls._nan

class MemoryObject(_MemoryObject):
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_1
            cls._instance = True
            cls.Team = Team2_1
            cls.memories = Memory2()

        return super().__new__(cls, *args, **kwargs)

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
    """fragment2 implement"""
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_2
            cls._instance = True
            cls.Team = Team2_2
            cls.memories = Memory2_1()

        return super().__new__(cls, *args, **kwargs)
    
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
            self.memoryCode = self.__class__.memories.append(state, reward, key)
            self.teamMemory = None
            return
        else:
            self.memoryCode = self.__class__.memories.choice()
            self.teamMemory = None
            return

class MemoryObject2(MemoryObject1):
    """operator implement"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_2
            cls._instance = True
            cls.Team = Team2_2
            cls.memories = Memory2_1()
        return super().__new__(cls, *args, **kwargs)

    def __add__(self, __o):       
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory + __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory + __o)
        else:
            return NotImplemented
    
    def __radd__(self, __o):
        if isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o + self.memory)
        else:
            return NotImplemented       
    
    def __sub__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory - __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory - __o)
        else:
            return NotImplemented
    
    def __rsub__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o - self.memory)
        else:
            return NotImplemented
    
    def __mul__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory * __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory * __o)
        else:
            return NotImplemented
    
    def __rmul__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o * self.memory)
        else:
            return NotImplemented
    
    def __pow__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory ** __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory ** __o)
        else:
            return NotImplemented
    
    def __rpow__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o ** self.memory)
        else:
            return NotImplemented
    
    def __truediv__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory / __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory / __o)
        else:
            return NotImplemented
    
    def __rtruediv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o / self.memory)
        else:
            return NotImplemented
    
    def __floordiv__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory // __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory // __o)
        else:
            return NotImplemented
    
    def __rfloordiv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o // self.memory)
        else:
            return NotImplemented
    
    def __mod__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory % __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory % __o)
        else:
            return NotImplemented
    
    def __rmod__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o % self.memory)
        else:
            return NotImplemented
    
    def __lt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.memory < __o.memory
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.memory < __o
        else:
            return NotImplemented
    
    def __rlt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return __o < self.memory
        else:
            return NotImplemented
    
    def __le__(self, __o):
        if isinstance(__o, self.__class__):
            return self.memory <= __o.memory
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.memory <= __o
        else:
            return NotImplemented
    
    def __rle__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return __o <= self.memory
        else:
            return NotImplemented

    def __lshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory << __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory << __o)
        else:
            return NotImplemented
    
    def __rlshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o << self.memory)
        else:
            return NotImplemented
    
    def __gt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.memory > __o.memory
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.memory > __o
        else:
            return NotImplemented
    
    def __rgt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return __o > self.memory
        else:
            return NotImplemented
    
    def __ge__(self, __o):
        if isinstance(__o, self.__class__):
            return self.memory >= __o.memory
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.memory >= __o
        else:
            return NotImplemented
    
    def __rge__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return __o >= self.memory
        else:
            return NotImplemented
    
    def __rshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory >> __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory >> __o)
        else:
            return NotImplemented
    
    def __rrshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o >> self.memory)
        else:
            return NotImplemented

    def __and__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory & __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory & __o)
        else:
            return NotImplemented

    def __rand__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory & __o)
        else:
            return NotImplemented
    
    def __or__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory | __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory | __o)
        else:
            return NotImplemented

    def __ror__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o | self.memory)
        else:
            return NotImplemented
            
    def __xor__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.memory ^ __o.memory)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(self.memory ^ __o)
        else:
            return NotImplemented
    
    def __rxor__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.memories.Fragment):
            return self.__class__(__o ^ self.memory)
        else:
            return NotImplemented
    
    def __invert__(self):
        return self.__class__(~self.memory)

    def __len__(self):
        return self.__class__.memories[self.memoryCode].size

    def getImage(self, _act, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_act, _state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}, {self.__class__}'
            self.__class__.memories.weights[self.memoryCode]*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()               # 忘却確立計上
            return self

class MemoryObject3(MemoryObject2):
    """operator implement"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_3
            cls._instance = True
            cls.Team = Team2_3
            cls.Memory = Memory3
            cls.memories = cls.Memory()
            cls._nan = cls.memories.nan
        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
            image = None,
            _task='task',
            initParams:dict or int =None
        ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(image, self.__class__.Team):
            self.teamMemory = image
            self.memoryCode = None
            return
        # The action is another action object
        elif isinstance(image, self.__class__):
            self.memoryCode = image.memoryCode
            self.teamMemory = image.teamMemory
            return
        elif isinstance(image, self.__class__.Memory.Fragment):
            assert image._id in self.__class__.memories, f'{image._id} not in {self.__class__.memories}'
            self.memoryCode = image._id
            self.teamMemory = None
            return
        elif isinstance(image, UUID):
            assert image in self.__class__.memories, f'{image} not in {self.__class__.memories}'
            self.memoryCode = image
            self.teamMemory = None
            return
        elif isinstance(image, list) or isinstance(image, np.ndarray):
            self.memoryCode = self.__class__.memories.append(image)
            self.teamMemory = None
            return
        else:
            try:
                self.memoryCode=self.__class__.memories.choices()[0]
                self.teamMemory=None
            except Exception as e:
                print('諦めな・・・', e)
                raise e

    def getImage(self, _state, visited, memVars, path_trace=None):
        if self.teamMemory is not None:
            return self.teamMemory.image(_state, visited, memVars=memVars, path_trace=path_trace)
        else:
            assert self.memoryCode in self.__class__.memories, f'{self.memoryCode} is not in {self.__class__.memories}, {self.__class__}'
            self.memory.weight*=0.9 # 忘却確立減算
            self.__class__.memories.updateWeights()              # 忘却確立計上
            return self

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pMemAtom=None, learner_id=None):
        if None in (mutateParams, parentTeam, teams, pMemAtom, learner_id):
            self.memoryCode=self.__class__.memories.choice(_ignore=[self.memoryCode])
            self.teamMemory=None
            return self

        if flip(pMemAtom):
            if self.memoryCode is not None:
                _ignore = self.memoryCode
            else:
                _ignore = None
            
            if not self.isAtomic():
                self.teamMemory.inLearners.remove(learner_id)
            
            self.memoryCode = self.__class__.memories.choice(_ignore=[_ignore])
            self.teamMemory = None
        else:
            selection_pool = [t for t in teams if t is not self.teamMemory and t is not parentTeam]
            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamMemory.inLearners.remove(learner_id)
                
                self.teamMemory = random.choice(selection_pool)
                self.teamMemory.inLearners.append(learner_id)

        return self

    @classmethod
    @property
    def NaN(cls):
        return cls(cls._nan)
    
class _ActionObject(_Logger):
    """
    Action  Object has a program to produce a value for the action, program doesn't
    run if just a discrete action code.
    """
    actions=[0]
    Team = None
    Memory=None
    # _instance = None
    # _logger  = None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import _Team
            cls._instance = True
            cls.Team = _Team
        return super().__new__(cls, *args, **kwargs)
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
                
    def __eq__(self, __o:object)->bool:
        '''
        An ActionObject is equal to another object if that object:
            - is an instance of the ActionObject class
            - has the same action code
            - has the same team action
        '''

        # The other object must be an instance of the ActionObject class
        if not isinstance(__o, self.__class__):    return False
        
        # The other object's action code must be equal to ours
        if self.actionCode != __o.actionCode:     return False
        
        # The other object's team action must be equal to ours
        if self.teamAction != __o.teamAction:     return False

        return True

    def __ne__(self, __o: object) -> bool:
        '''
        Negate __eq__
        '''
        return not self.__eq__(__o)

    def __str__(self):
        return f"TeamAction {self.teamAction} ActionCode: {self.actionCode}"

    def __repr__(self):
        return f"TeamAction {self.teamAction} ActionCode: {self.actionCode}"

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
    actions=None
    _nan=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_1
            cls._instance = True
            cls.Team = Team1_1
            cls.actions = Memory1()
            
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

    def __str__(self):
        if self.actionCode:
            return f"code:{self.actionCode}, action:{self.action}"
        return f"team:{self.teamAction}"

    def __repr__(self):
        if self.actionCode:
            return f"code:{self.actionCode}, action:{self.action}"
        return f"team:{self.teamAction}"

    def __getitem__(self, _key):
        return self.__class__.actions[self.actionCode][_key]
    
    def getAction(self, _state, visited, actVars, path_trace=None):
        if self.teamAction is not None:
            return self.teamAction.act(_state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # breakpoint(self.__class__, __class__, self.__class__.NaN) # (Any, ActionObj1, property object)
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

    @classmethod
    @property
    def NaN(cls):
        return cls._nan

    @property
    def action(self):
        return self.__class__.actions[self.actionCode]

class ActionObject2(ActionObject1):
    """operator implement"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_2
            cls._instance = True
            cls.Team = Team1_2
            cls.Memory = Memory3
            cls.actions = cls.Memory()

        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
        action = None,
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
            # self.debug(f'init:teamObj, {self}')
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            # breakpoint(self.__class__)
            # self.debug(f'init:actObj, {self}')
            return
        # An int means the action is an index into the action codes in initParams
        elif isinstance(action, UUID):
            assert action in self.__class__.actions, f'{action} not in {self.__class__.actions}'
            self.actionCode = action
            self.teamAction = None
            # self.debug(f'init:UUID, {self}')
            return
        elif isinstance(action, list) or isinstance(action, np.ndarray):
            self.actionCode = self.__class__.actions.append(action)
            self.teamAction = None
            # self.debug(f'init:list, {self}')
            return
        else:
            try:
                # self.debug(f'init:list, actObj:{self}, action:{self.action}')
                self.actionCode=self.__class__.actions.choice()
                self.teamAction=None
                # self.debug(f'init:choice, {self}')
            except:
                print('諦めな・・・')

    def __add__(self, __o):       
        if isinstance(__o, self.__class__):
            return self.__class__(self.action + __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action + __o)
        else:
            return NotImplemented
    
    def __radd__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o + self.action)
        else:
            return NotImplemented       
    
    def __sub__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action - __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action - __o)
        else:
            return NotImplemented
    
    def __rsub__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o - self.action)
        else:
            return NotImplemented
    
    def __mul__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action * __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action * __o)
        else:
            return NotImplemented
    
    def __rmul__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o * self.action)
        else:
            return NotImplemented
    
    def __pow__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action ** __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action ** __o)
        else:
            return NotImplemented
    
    def __rpow__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o ** self.action)
        else:
            return NotImplemented
    
    def __truediv__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action / __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action / __o)
        else:
            return NotImplemented
    
    def __rtruediv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o / self.action)
        else:
            return NotImplemented
    
    def __floordiv__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action // __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action // __o)
        else:
            return NotImplemented
    
    def __rfloordiv__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o // self.action)
        else:
            return NotImplemented
    
    def __mod__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action % __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action % __o)
        else:
            return NotImplemented
    
    def __rmod__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o % self.action)
        else:
            return NotImplemented
    
    def __lt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.action < __o.action
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.action < __o
        else:
            return NotImplemented
    
    def __rlt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return __o < self.action
        else:
            return NotImplemented
    
    def __le__(self, __o):
        if isinstance(__o, self.__class__):
            return self.action <= __o.action
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.action <= __o
        else:
            return NotImplemented
    
    def __rle__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return __o <= self.action
        else:
            return NotImplemented

    def __lshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action << __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action << __o)
        else:
            return NotImplemented
    
    def __rlshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o << self.action)
        else:
            return NotImplemented
    
    def __gt__(self, __o):
        if isinstance(__o, self.__class__):
            return self.action > __o.action
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.action > __o
        else:
            return NotImplemented
    
    def __rgt__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return __o > self.action
        else:
            return NotImplemented
    
    def __ge__(self, __o):
        if isinstance(__o, self.__class__):
            return self.action >= __o.action
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.action >= __o
        else:
            return NotImplemented
    
    def __rge__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return __o >= self.action
        else:
            return NotImplemented
    
    def __rshift__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action >> __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action >> __o)
        else:
            return NotImplemented
    
    def __rrshift__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o >> self.action)
        else:
            return NotImplemented

    def __and__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action & __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action & __o)
        else:
            return NotImplemented

    def __rand__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action & __o)
        else:
            return NotImplemented
    
    def __or__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action | __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action | __o)
        else:
            return NotImplemented

    def __ror__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o | self.action)
        else:
            return NotImplemented
            
    def __xor__(self, __o):
        if isinstance(__o, self.__class__):
            return self.__class__(self.action ^ __o.action)
        elif isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(self.action ^ __o)
        else:
            return NotImplemented
    
    def __rxor__(self, __o):
        if isinstance(__o, np.ndarray) or isinstance(__o, list) \
            or isinstance(__o, int) or isinstance(__o, float) \
            or isinstance(__o, self.__class__.actions.Fragment):
            return self.__class__(__o ^ self.action)
        else:
            return NotImplemented
    
    def __invert__(self):
        return self.__class__(~self.action)

    def getAction(self, _state, visited, actVars, path_trace=None):
        if self.teamAction is not None:
            return self.teamAction.act(_state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # breakpoint(self.__class__, __class__, self.__class__.NaN) # (Any, ActionObj1, property object)
            assert self.actionCode in self.__class__.actions, f'{self.actionCode} is not in {self.__class__.actions}'
            self.__class__.actions[self.actionCode].weight*=0.9 # 忘却確立減算
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
                self.teamAction.inLearners.remove(learner_id)

            self.actionCode = self.__class__.actions.choice([_ignore])
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(learner_id)

                self.teamAction = random.choice(selection_pool)
                self.teamAction.inLearners.append(learner_id)
       
        return self
       
    @property
    def action(self):
        return self.actions[self.actionCode].fragment

    @property
    def bid(self):
        return self.action[0]

    @property
    def ch(self):
        return int(self.action[-1]//1)
    
    @property
    def weight(self):
        return self.actions[self.actionCode].weight
    
    @classmethod
    @property
    def NaN(cls):
        return cls(cls._nan)

class ActionObject3(ActionObject2):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_2_1
            cls._instance = True
            cls.Team = Team1_2_1
            cls.Memory = Memory3_1
            cls.actions = cls.Memory()

        return super().__new__(cls, *args, **kwargs)

class ActionObject4(ActionObject3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_2_2
            cls._instance = True
            cls.Team = Team1_2_2
            cls.Memory = Memory3_2
            cls.actions = cls.Memory()

        return super().__new__(cls, *args, **kwargs)

class ActionObject5(ActionObject4):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_2_3
            cls._instance = True
            cls.Team = Team1_2_3
            cls.Memory = Memory3_2
            cls.actions = cls.Memory()

        return super().__new__(cls, *args, **kwargs)

    
    def getAction(self, _state, visited, actVars, path_trace=None):
        if self.teamAction is not None:
            return self.teamAction.act(_state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # breakpoint(self.__class__, __class__, self.__class__.NaN) # (Any, ActionObj1, property object)
            assert self.actionCode in self.__class__.actions, f'{self.actionCode} is not in {self.__class__.actions}'
            self.__class__.actions[self.actionCode].weight*=0.9 # 忘却確立減算
            self.__class__.actions.updateWeights()               # 忘却確立計上
            return self


class Qualia(ActionObject2):
    """ Memory object management

        より一致度の高いクオリアを生成することが目的。
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_3_1
            cls._instance = True
            cls.Team = Team1_3_1
            cls.Memory = Memory3
            cls.actions = cls.Memory()
            cls._nan = cls.actions.nan

        return super().__new__(cls, *args, **kwargs)

class Operation(ActionObject2):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_3_2
            cls._instance = True
            cls.Team = Team1_3_2
            cls.Memory = Memory3
            cls.actions = cls.Memory(GAMMA)
            cls._nan = cls.actions.nan

        return super().__new__(cls, *args, **kwargs)

class Hippocampus(MemoryObject3):
    """ Short term memory management

        呼び出し回数の多いクオリアを生成することが目的
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team2_3
            cls._instance = True
            cls.Team = Team2_3
            cls.Memory = Memory3
            cls.memories = cls.Memory()
            cls._nan = cls.memories.nan
        return super().__new__(cls, *args, **kwargs)
