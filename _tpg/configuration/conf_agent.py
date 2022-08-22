from math import tanh
from importlib.metadata import distribution
import pickle
from random import random
import time
import numpy as np
from _tpg.action_object import ActionObject
from _tpg.memory_object import MemoryObject

"""
Simplified wrapper around a (root) team for easier interface for user.
"""
class ConfAgent:

    """
    Create an agent with a team.
    """
    def init_def(self, team, functionsDict, num=1, actVars=None):
        self.team = team
        self.functionsDict = functionsDict
        self.agentNum = num
        self.actVars = actVars

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act_def(self, state, path_trace=None):

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
    """
    Give this agent/root team a reward for the given task
    """
    def reward_def(self, score, task='task'):
        self.team.outcomes[task] = score

    """
    Check if agent completed this task already, to skip.
    """
    def taskDone_def(self, task):
        return task in self.team.outcomes

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile_def(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))

class ConfAgent1:

    """
    Create an agent with a team.
    """
    def init_def(self, team, functionsDict, num=1, actVars=None):
        self.team = team
        self.functionsDict = functionsDict
        self.agentNum = num
        self.actVars = actVars

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act_def(self, state, path_trace=None):

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
    """
    Give this agent/root team a reward for the given task
    """
    def reward_def(self, score, task='task'):
        self.team.outcomes[task] = score

    """
    Check if agent completed this task already, to skip.
    """
    def taskDone_def(self, task):
        return task in self.team.outcomes

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile_def(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))

class ConfAgent2:

    def init_def(self, team, functionsDict, num=1, actVars=None):
        self.team = team
        self.functionsDict = functionsDict
        self.agentNum = num
        self.actVars = actVars
        self.imageCode = None


    def image_def(self, _act, _state, path_trace=None):

        start_execution_time = time.time()*1000.0
        self.actVars["frameNum"] = random()
        visited = list() #Create a new list to track visited team/learners each time
        
        path = None
        if path_trace != None:
            path = list()
            imageCode = self.team.image(_act, _state, visited=visited, actVars=self.actVars, path_trace=path)
        else:
            imageCode = self.team.image(_act, _state, visited=visited, actVars=self.actVars)

        end_execution_time = time.time()*1000.0
        execution_time = end_execution_time - start_execution_time

        # state長の制限が存在する。 その場合Rejectされるべき？ 上限以上切り捨てで出力
        # state = MemoryObject.memories[imageCode].recall(_state)
        if path_trace != None:
            path_trace['execution_time'] = execution_time
            path_trace['execution_time_units'] = 'milliseconds'
            path_trace['root_team_id'] = str(self.team.id)
            path_trace['final_image'] = _state
            path_trace['path'] = path 
            path_trace['depth'] = len(path)

            
        return imageCode

    def reward_def(self, score, task='task'):
        if not self.team.outcomes.get(task): self.team.outcomes[task]=0.
        # distribution = self.team.numDistribution()
        # distribute_score = score/float(distribution)
        # survive_rate = tanh(distribute_score*3)+self.team.outcomes['survive']
        self.team.outcomes[task]=score
        # self.team.outcomes[task] = tanh(task_score*3)
        # self.team.outcomes['reward'] = distribute_score
        # self.team.outcomes['survive'] = tanh(survive_rate*3)
        # inheritance team counts を導入することで、報酬分配を考えることができる。
        # このteamを継承しているteamにも報酬を支払う。 分け与える

    def taskDone_def(self, task):
        return task in self.team.outcomes

    def saveToFile_def(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))

class ConfAgent3:

    """
    Create an agent with a team.
    """
    def init_def(self, team, functionsDict, num=1, actVars={'task':'task'}):
        self.team = team
        self.functionsDict = functionsDict
        self.agentNum = num
        self.actVars = actVars # has 'task'

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act_def(self, state, path_trace=None):

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
    """
    Give this agent/root team a reward for the given task
    """
    def reward_def(self, score, task='task'):
        # distribution = self.team.numDistribution()
        # distribute_score = score/float(distribution)
        # survive_rate = tanh(distribute_score*3)+self.team.outcomes['survive']
        # self.team.outcomes[task] = tanh(distribute_score*3)
        # self.team.outcomes['reward'] = distribute_score
        # self.team.outcomes['survive'] = tanh(survive_rate*3)
        # assert isinstance(self.team.outcomes[task], float), type(float(self.team.outcomes[task]))
        self.team.outcomes[task]=score


    """
    Check if agent completed this task already, to skip.
    """
    def taskDone_def(self, task):
        return task in self.team.outcomes

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile_def(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))
