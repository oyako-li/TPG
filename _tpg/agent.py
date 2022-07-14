# from program import Program
import pickle
from random import random
import time


def loadAgent(fileName):
    agent = pickle.load(open(fileName, 'rb'))
    agent.configFunctionsSelf()
    return agent

class Agent:

    """
    Create an agent with a team.
    """
    def __init__(self, team, functionsDict:dict, num:int=1, actVars:dict=None):
        self.team = team
        self.functionsDict:dict = functionsDict
        self.agentNum:int = num
        self.actVars:dict = actVars
        # logger.debug(f'{__name__}/Agent.init()')

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

        # logger.debug(f'{__name__}/Atgent.act->{result}')
            
        return result

    """
    Give this agent/root team a reward for the given task
    """
    def reward(self, score, task='task'):
        self.team.outcomes[task] = score

    """
    Check if agent completed this task already, to skip.
    """
    def taskDone(self, task):
        return task in self.team.outcomes

    def zeroRegisters(self):
        self.team.zeroRegisters()

    """
    Should be called when the agent is loaded from a file or when loaded into 
    another process/thread, to ensure proper function used in all classes.
    """
    def configFunctionsSelf(self):
        from _tpg.team import Team
        from _tpg.learner import Learner
        from _tpg.action_object import ActionObject
        from _tpg.program import Program

        # first set up Agent functions
        Agent.configFunctions(self.functionsDict["Agent"])

        # set up Team functions
        Team.configFunctions(self.functionsDict["Team"])

        # set up Learner functions
        Learner.configFunctions(self.functionsDict["Learner"])

        # set up ActionObject functions
        ActionObject.configFunctions(self.functionsDict["ActionObject"])

        # set up Program functions
        Program.configFunctions(self.functionsDict["Program"])

    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_agent import ConfAgent

        if functionsDict["init"] == "def":
            cls.__init__ = ConfAgent.init_def

        if functionsDict["act"] == "def":
            cls.act = ConfAgent.act_def

        if functionsDict["reward"] == "def":
            cls.reward = ConfAgent.reward_def

        if functionsDict["taskDone"] == "def":
            cls.taskDone = ConfAgent.taskDone_def

        if functionsDict["saveToFile"] == "def":
            cls.saveToFile = ConfAgent.saveToFile_def

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile(self, fileName):
        pickle.dump(self, open(fileName, 'wb'))

class Agent1:

    """
    Create an agent with a team.
    """
    def __init__(self, team, functionsDict:dict, num:int=1, actVars:dict=None)->None: pass

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act(self, state, path_trace=None): pass

    """
    Give this agent/root team a reward for the given task
    """
    def reward(self, score, task='task')->None: pass

    """
    Check if agent completed this task already, to skip.
    """
    def taskDone(self, task): pass

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile(self, fileName): pass

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    """
    Should be called when the agent is loaded from a file or when loaded into 
    another process/thread, to ensure proper function used in all classes.
    """
    def configFunctionsSelf(self)->None:
        from _tpg.team import Team1
        from _tpg.learner import Learner1
        from _tpg.program import Program1
        from _tpg.action_object import ActionObject1

        # first set up Agent functions
        Agent1.configFunctions(self.functionsDict["Agent"])

        # set up Team functions
        Team1.configFunctions(self.functionsDict["Team"])

        # set up Learner functions
        Learner1.configFunctions(self.functionsDict["Learner"])

        # set up ActionObject functions
        ActionObject1.configFunctions(self.functionsDict["ActionObject"])

        # set up Program functions
        Program1.configFunctions(self.functionsDict["Program"])

    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_agent import ConfAgent1

        if functionsDict["init"] == "def":
            cls.__init__ = ConfAgent1.init_def

        if functionsDict["act"] == "def":
            cls.act = ConfAgent1.act_def

        if functionsDict["reward"] == "def":
            cls.reward = ConfAgent1.reward_def

        if functionsDict["taskDone"] == "def":
            cls.taskDone = ConfAgent1.taskDone_def

        if functionsDict["saveToFile"] == "def":
            cls.saveToFile = ConfAgent1.saveToFile_def

class Agent2:

    def __init__(self, team, functionsDict:dict, num:int=1, actVars:dict=None)->None: pass

    def image(self, _act, _state, path_trace=None): pass

    def reward(self, state, task='task')->None: pass

    def taskDone(self, task): pass

    def saveToFile(self, fileName): pass

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    def configFunctionsSelf(self)->None:
        from _tpg.team import Team2
        from _tpg.learner import Learner2
        from _tpg.program import Program2
        # from _tpg.action_object import ActionObject2

        # first set up Agent functions
        Agent2.configFunctions(self.functionsDict["Agent"])

        # set up Team functions
        Team2.configFunctions(self.functionsDict["Team"])

        # set up Learner functions
        Learner2.configFunctions(self.functionsDict["Learner"])

        # set up ActionObject functions
        # ActionObject2.configFunctions(self.functionsDict["ActionObject"])

        # set up Program functions
        Program2.configFunctions(self.functionsDict["Program"])

    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_agent import ConfAgent2

        if functionsDict["init"] == "def":
            cls.__init__ = ConfAgent2.init_def

        if functionsDict["image"] == "def":
            cls.image = ConfAgent2.image_def

        if functionsDict["reward"] == "def":
            cls.reward = ConfAgent2.reward_def

        if functionsDict["taskDone"] == "def":
            cls.taskDone = ConfAgent2.taskDone_def

        if functionsDict["saveToFile"] == "def":
            cls.saveToFile = ConfAgent2.saveToFile_def

class Agent3:

    """
    Create an agent with a team.
    """
    def __init__(self, team, functionsDict:dict, num:int=1, actVars:dict=None)->None: pass

    """
    Gets an action from the root team of this agent / this agent.
    """
    def act(self, state, path_trace=None): pass

    """
    Give this agent/root team a reward for the given task
    """
    def reward(self, score, task='task')->None: pass

    """
    Check if agent completed this task already, to skip.
    """
    def taskDone(self, task): pass

    """
    Save the agent to the file, saving any relevant class values to the instance.
    """
    def saveToFile(self, fileName): pass

    def zeroRegisters(self)->None:
        self.team.zeroRegisters()

    """
    Should be called when the agent is loaded from a file or when loaded into 
    another process/thread, to ensure proper function used in all classes.
    """
    def configFunctionsSelf(self)->None:
        from _tpg.team import Team3
        from _tpg.learner import Learner3
        from _tpg.program import Program3
        from _tpg.action_object import ActionObject3

        # first set up Agent functions
        Agent3.configFunctions(self.functionsDict["Agent"])

        # set up Team functions
        Team3.configFunctions(self.functionsDict["Team"])

        # set up Learner functions
        Learner3.configFunctions(self.functionsDict["Learner"])

        # set up ActionObject functions
        ActionObject3.configFunctions(self.functionsDict["ActionObject"])

        # set up Program functions
        Program3.configFunctions(self.functionsDict["Program"])

    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_agent import ConfAgent3

        if functionsDict["init"] == "def":
            cls.__init__ = ConfAgent3.init_def

        if functionsDict["act"] == "def":
            cls.act = ConfAgent3.act_def

        if functionsDict["reward"] == "def":
            cls.reward = ConfAgent3.reward_def

        if functionsDict["taskDone"] == "def":
            cls.taskDone = ConfAgent3.taskDone_def

        if functionsDict["saveToFile"] == "def":
            cls.saveToFile = ConfAgent3.saveToFile_def
