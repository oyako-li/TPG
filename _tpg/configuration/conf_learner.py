from _tpg.program import Program
from _tpg.action_object import ActionObject
from _tpg.team import Team1
from _tpg.utils import flip
import numpy as np
import random
import time
import copy
import uuid

"""
A team has multiple learners, each learner has a program which is executed to
produce the bid value for this learner's action.
"""
class ConfLearner:

    """
    Create a new learner, either copied from the original or from a program or
    action. Either requires a learner, or a program/action pair.
    """
    def init_def(self, initParams, program, actionObj, numRegisters, learner_id=None):
        self.program = Program(
            instructions=program.instructions
        ) #Each learner should have their own copy of the program
        self.actionObj = ActionObject(action=actionObj, initParams=initParams) #Each learner should have their own copy of the action object
        self.registers = np.zeros(numRegisters, dtype=float)

        self.ancestor = None #By default no ancestor

        '''
        TODO What's self.states? 
        '''
        self.states = []


        self.inTeams = [] # Store a list of teams that reference this learner, incoming edges
        

        self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        '''
        TODO should this be -1 before it sees any frames?
        '''
        self.frameNum = 0 # Last seen frame is 0

        # Assign id from initParams counter
        self.id = uuid.uuid4()


        if not self.isActionAtomic():
            self.actionObj.teamAction.inLearners.append(str(self.id))

        #print("Creating a brand new learner" if learner_id == None else "Creating a learner from {}".format(str(learner_id)))
        #print("Created learner {} [{}] -> {}".format(self.id, "atomic" if self.isActionAtomic() else "Team", self.actionObj.actionCode if self.isActionAtomic() else self.actionObj.teamAction.id))
        

    """
    Get the bid value, highest gets its action selected.
    """
    def bid_def(self, state, actVars=None):
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[0]

    """
    Get the bid value, highest gets its action selected. Passes memory args to program.
    """
    def bid_mem(self, state, actVars=None):
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        Program.memWriteProbFunc)

        return self.registers[0]

    """
    Returns the action of this learner, either atomic, or requests the action
    from the action team.
    """
    def getAction_def(self, state, visited, actVars=None, path_trace=None):
        return self.actionObj.getAction(state, visited, actVars=actVars, path_trace=path_trace)



    """
    Gets the team that is the action of the learners action object.
    """
    def getActionTeam_def(self):
        return self.actionObj.teamAction

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isActionAtomic_def(self):
        return self.actionObj.isAtomic()

    """
    Mutates either the program or the action or both.
    """
    def mutate_def(self, mutateParams, parentTeam, teams, pActAtom):

        changed = False
        while not changed:
            # mutate the program
            if flip(mutateParams["pProgMut"]):

                changed = True
              
                self.program.mutate(mutateParams)

            # mutate the action
            if flip(mutateParams["pActMut"]):

                changed = True
                
                self.actionObj.mutate(mutateParams, parentTeam, teams, pActAtom, learner_id=self.id)

        return self

from _tpg.program import Program1
from _tpg.action_object import ActionObject1


class ConfLearner1:
    def init_def(self, 
        initParams:int or dict, 
        program:Program1, 
        actionObj:Team1 or ActionObject1 or int, 
        numRegisters:int or np.ndarray, 
        _ancestor=None,
        _states:list=[],
        _inTeams:list=[],
        _frameNum:int=0
    ):
        self.program = Program1(
            instructions=program.instructions
        ) #Each learner should have their own copy of the program
        self.actionObj = ActionObject1(
            action=actionObj, 
            initParams=initParams
        ) #Each learner should have their own copy of the action object
        if isinstance(numRegisters, int): self.registers = np.zeros(numRegisters, dtype=float) # 子供に記憶は継承されない。
        else: self.registers = numRegisters
        if isinstance(initParams, int): self.genCreate = initParams # Store the generation that this learner was created on
        elif isinstance(initParams, dict): self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        self.ancestor = _ancestor #By default no ancestor
        self.states = _states
        self.inTeams = _inTeams # Store a list of teams that reference this learner, incoming edges
        # self.actionCodes = initParams["actionCodes"]
        self.frameNum = _frameNum # Last seen frame is 0
        self.id = uuid.uuid4()


        if not self.isActionAtomic(): self.actionObj.teamAction.inLearners.append(str(self.id))

    def bid_def(self, state, actVars=None):
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        Program1.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[0]

    def bid_mem(self, state, actVars=None):
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        Program1.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        Program1.memWriteProbFunc)

        return self.registers[0]

    def getAction_def(self, state, visited, actVars=None, path_trace=None):
        return self.actionObj.getAction(state, visited, actVars=actVars, path_trace=path_trace)

    def getActionTeam_def(self):
        return self.actionObj.teamAction

    def isActionAtomic_def(self):
        return self.actionObj.isAtomic()

    def mutate_def(self, mutateParams, parentTeam, teams, pActAtom):

        changed = False
        while not changed:
            # mutate the program
            if flip(mutateParams["pProgMut"]):

                changed = True
              
                self.program.mutate(mutateParams)

            # mutate the action
            if flip(mutateParams["pActMut"]):

                changed = True
                
                self.actionObj.mutate(mutateParams, parentTeam, teams, pActAtom, learner_id=self.id)

        return self

    def clone_def(self):
        # _clone = self.__init__(initParams={"generation":self.genCreate}, program=self.program, actionObj=self.actionObj, numRegisters=len(self.registers))
        # _clone.registers = self.registers
        # _clone.ancestor = self.ancestor
        # _clone.states = self.states
        # # _clone.inTeams = self.inTeams # ここが変わるはず
        # _clone.frameNum = self.frameNum
        return self