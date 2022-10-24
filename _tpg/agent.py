# from program import Program
from math import tanh
import pickle
from random import random
import time

class _Agent:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls)

    def __init__(self, team, num:int=1, actVars:dict=None)->None:
        """
        Create an agent with a team.
        """
        self.team = team
        self.agentNum = num
        self.actVars = actVars

    def __hash__(self):
        return int(self.team.id)

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
            path_trace['root_team_id'] = str(self.team.id)
            path_trace['final_action'] = result
            path_trace['path'] = path 
            path_trace['depth'] = len(path)
            
        return result

    def reward(self, score=0, task='task'):
        """
        Give this agent/root team a reward for the given task
        """
        self.team[task] = score

    def taskDone(self, task):
        """
        Check if agent completed this task already, to skip.
        """
        return task in self.team.outcomes
    
    @property
    def id(self):
        return str(self.team.id)
    
    def saveToFile_def(self, fileName):
        """
        Save the agent to the file, saving any relevant class values to the instance.
        """
        pickle.dump(self, open(fileName, 'wb'))

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    @classmethod
    def loadAgent(cls, fileName):
        agent = pickle.load(open(fileName, 'rb'))
        assert(isinstance(agent, cls), 'this file is different Class type')
        return agent

class Agent1(_Agent):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

class Agent1_1(Agent1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    
    def act(self, state, path_trace=None): 
        """
        Gets an action from the root team of this agent / this agent.
        """
        start_execution_time = time.time()*1000.0
        self.actVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        if not self.team.outcomes.get(self.actVars['task']): self.team.outcomes[self.actVars['task']]=0.

        
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
            path_trace['root_team_id'] = str(self.team.id)
            path_trace['final_action'] = result
            path_trace['path'] = path 
            path_trace['depth'] = len(path)
            
        return result

    
    def reward(self, score=0, task='task'):
        self.team[task] += tanh(score)
        self.team[task] = tanh(self.team[task])

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

    def image(self, act, state, path_trace=None): 
        """
        Gets an action from the root team of this agent / this agent.
        act = int or actionObject,
        state = np.ndarray or memoryObject
        """
        start_execution_time = time.time()*1000.0
        self.memVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        if not self.team.outcomes.get(self.memVars['task']): self.team.outcomes[self.memVars['task']]=0.
        
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
    
    def reward(self, score=0, task='task'):
        self.team[task] += tanh(score)
        self.team[task] = tanh(self.team[task])