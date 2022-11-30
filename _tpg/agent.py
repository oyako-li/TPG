# from program import Program
from math import tanh
from datetime import datetime
from _tpg.utils import _Logger, sigmoid, sigmoid2
import pickle
from random import random
import time


class _Agent(_Logger):
    # _instance = None
    # _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, team, num:int=1, actVars:dict=None)->None:
        """
        Create an agent with a team.
        """
        self.team = team
        self.agentNum = num
        self.actVars = actVars
        self.score = 0.
        self.sequence=list()

    def __hash__(self):
        return int(self.id)

    def act(self, state, path_trace=None): 
        """
        Gets an action from the root team of this agent / this agent.
        """
        start_execution_time = time.time()*1000.0
        self.actVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        
        result = None
        path = None
        if path_trace != None:
            path = list()
            result = self.team.act(state, visited=visited, actVars=self.actVars, path_trace=path)
        else:
            result = self.team.act(state, visited=visited, actVars=self.actVars)

        end_execution_time = time.time()*1000.0
        execution_time = end_execution_time - start_execution_time
        if path_trace != None:

            path_trace['execution_time'] = execution_time
            path_trace['execution_time_units'] = 'milliseconds'
            path_trace['root_team_id'] = self.id
            path_trace['final_action'] = result
            path_trace['path'] = path 
            path_trace['depth'] = len(path)
        # self.info(f'evolving:{datetime.now()}')
        
        return result

    def reward(self, score=None, task='task'):
        """
        Give this agent/root team a reward for the given task
        """
        self.info(f'task:{task}, agent_id:{self.id}, score:{self.score}')
        _score = score if score else self.score
        self.team[task] += sigmoid2(_score)

    def taskDone(self, task):
        """
        Check if agent completed this task already, to skip.
        """
        return task in self.team.outcomes
  
    def saveToFile_def(self, fileName):
        """
        Save the agent to the file, saving any relevant class values to the instance.
        """
        pickle.dump(self, open(fileName, 'wb'))

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    def trace(self, _sequence):
        # assert _sequence != [], f'{_sequence} should not non list'
        if _sequence == []: return

        self.team.sequence = list(_sequence)
        self.debug(f'trace_sequence:{self.team.sequence},')

    @property
    def id(self):
        return str(self.team.id)
    
    @classmethod
    def loadAgent(cls, fileName):
        agent = pickle.load(open(fileName, 'rb'))
        assert isinstance(agent, cls), 'this file is different Class type'
        return agent

class Agent1(_Agent):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    @property
    def id(self):
        return self.team.id

class Agent1_1(Agent1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

class Agent1_2(Agent1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def trace(self, _sequence):
        # assert _sequence != [], f'{_sequence} should not non list'
        if _sequence == []: return

        self.team.appendSequence(_sequence)
        self.debug(f'trace_sequence:{self.team.sequence},')


class Agent1_3(Agent1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)


class Agent2(_Agent):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, team, num:int=1, memVars:dict=None)->None:
        """
        Create an agent with a team.
        """
        self.team = team
        self.agentNum = num
        self.memVars = memVars
        self.score = 0.

    def image(self, act, state, path_trace=None): 
        """
        Gets an action from the root team of this agent / this agent.
        act = int or actionObject,
        state = np.ndarray or memoryObject
        """
        start_execution_time = time.time()*1000.0
        self.memVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        
        result = None
        path = None
        if path_trace != None:
            path = list()
            result = self.team.image(act, state, visited=visited, memVars=self.memVars, path_trace=path)
        else:
            result = self.team.image(act, state, visited=visited, memVars=self.memVars)

        end_execution_time = time.time()*1000.0
        execution_time = end_execution_time - start_execution_time
        if path_trace != None:

            path_trace['execution_time'] = execution_time
            path_trace['execution_time_units'] = 'milliseconds'
            path_trace['root_team_id'] = self.id
            path_trace['final_image'] = result
            path_trace['path'] = path 
            path_trace['depth'] = len(path)
            
        return result

class Agent2_1(Agent2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)
   
    def reward(self, score=0, task='task'):
        if not self.team.outcomes.get(task):
            self.team.outcomes[task]=0.

        score = score if score else self.score
        self.team[task] += tanh(score)
        # self.team[task] = tanh(self.team[task])

class Agent2_3(Agent2):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, team, num:int=1, memVars:dict=None)->None:
        """
        Create an agent with a team.
        """
        self.team = team
        self.agentNum = num
        self.memVars = memVars
        self.score = 0.

    def image(self, state, path_trace=None): 
        """
        Gets an action from the root team of this agent / this agent.
        act = int or actionObject,
        state = np.ndarray or memoryObject
        """
        start_execution_time = time.time()*1000.0
        self.memVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        

        result = None
        path = None
        if path_trace != None:
            path = list()
            result = self.team.image(state, visited=visited, memVars=self.memVars, path_trace=path)
        else:
            result = self.team.image(state, visited=visited, memVars=self.memVars)

        end_execution_time = time.time()*1000.0
        execution_time = end_execution_time - start_execution_time
        if path_trace != None:

            path_trace['execution_time'] = execution_time
            path_trace['execution_time_units'] = 'milliseconds'
            path_trace['root_team_id'] = self.id
            path_trace['final_image'] = result
            path_trace['path'] = path 
            path_trace['depth'] = len(path)
            
        return result

    def reward(self, score=None, task='task'):
        if not self.team.outcomes.get(task):
            self.team.outcomes[task]=0.

        score = score if score else self.score
        # self.team[task] += tanh(score)
        self.team[task] += sigmoid(score)
        # self.team[task] = tanh(self.team[task])

    def trace(self, _sequence):
        self.team.sequence = list(_sequence)

    @property
    def id(self):
        return self.team.id