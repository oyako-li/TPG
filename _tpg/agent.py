# from program import Program
import pickle
from random import random
import time

class _Agent:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance
    """
    Create an agent with a team.
    """
    def __init__(self, team, num:int=1, actVars:dict=None)->None:
        self.team = team
        self.agentNum = num
        self.actVars = actVars

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act(self, state, path_trace=None): 
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
    def reward(self, score, task='task')->None:
        self.team.outcomes[task] = score


    """
    Check if agent completed this task already, to skip.
    """
    def taskDone(self, task):
        return task in self.team.outcomes
        

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile_def(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    @classmethod
    def loadAgent(cls, fileName):
        agent = pickle.load(open(fileName, 'rb'))
        assert(isinstance(agent, cls), 'this file is different Class type')
        return agent
