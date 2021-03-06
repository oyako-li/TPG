from _tpg.program import Program, Program1, Program3, Program2
from _tpg.action_object import ActionObject, ActionObject1, ActionObject3, ActionObject2
from _tpg.utils import flip
import numpy as np
import random

"""
Action  Object has a program to produce a value for the action, program doesn't
run if just a discrete action code.
"""
class ConfActionObject:

    def init_def(self, initParams=None, action = None):

        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''
        from _tpg.team import Team

        # The action is a team
        if isinstance(action, Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
    

        # The action is another action object
        if isinstance(action, ActionObject):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return

        # An int means the action is an index into the action codes in initParams
        if isinstance(action, int):
            if initParams is not None:
                if "actionCodes" not in initParams:
                    raise Exception('action codes not found in init params', initParams)

                try:
                    ActionObject._actions = initParams["actionCodes"]
                    self.actionCode = initParams["actionCodes"][action]
                    self.teamAction = None
                except IndexError as err:
                    '''
                    TODO log index error
                    '''
                    print("Index error")
                return
            else:
                try:
                    self.actionCode=random.choice(ActionObject._actions)
                    self.teamAction=None
                except:
                    print('諦めな・・・')
                return

    def init_real(self, initParams=None, action=None):

        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''
        from team import Team

        
        if isinstance(action, Team):
            # The action is a team
            self.actionCode = None
            self.actionLength = None
            self.teamAction = action
            self.program = Program(initParams=initParams, 
                    maxProgramLength=initParams["initMaxActProgSize"],
                    nOperations=initParams["nOperations"],
                    nDestinations=initParams["nDestinations"],
                    inputSize=initParams["inputSize"])

        elif isinstance(action, ActionObject):
            # The action is another action object
            self.actionCode = action.actionCode
            self.actionLength = action.actionLength
            self.teamAction = action.teamAction
            self.program = Program(instructions=action.program.instructions,
                                    initParams=initParams)

        elif isinstance(action, int):
            # An int means the action is an index into the action codes in initParams
            
            if "actionCodes" not in initParams:
                raise Exception('action codes not found in init params', initParams)

            try:
                ActionObject._actions = initParams["actionCodes"]

                self.actionCode = initParams["actionCodes"][action]
                self.actionLength = initParams["actionLengths"][action]
                self.teamAction = None
                self.program = Program(initParams=initParams, 
                    maxProgramLength=initParams["initMaxActProgSize"],
                    nOperations=initParams["nOperations"],
                    nDestinations=initParams["nDestinations"],
                    inputSize=initParams["inputSize"])
            except IndexError as err:
                '''
                TODO log index error
                '''
                print("Index error")

        self.registers = np.zeros(max(initParams["nActRegisters"], initParams["nDestinations"]))

    """
    Returns the action code, and if applicable corresponding real action.
    """
    def getAction_def(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    """
    Returns the action code, and if applicable corresponding real action(s).
    """
    def getAction_real(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            if self.actionLength == 0:
                return self.actionCode, None
            else:
                return self.actionCode, self.getRealAction(state, actVars=actVars)

    """
    Gets the real action from a register.
    """
    def getRealAction_real(self, state, actVars=None):
        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[:self.actionLength]

    """
    Gets the real action from a register. With memory.
    """
    def getRealAction_real_mem(self, state, actVars=None):
        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        Program.memWriteProbFunc)

        return self.registers[:self.actionLength]

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isAtomic_def(self):
        return self.teamAction is None

    """
    Change action to team or atomic action.
    """
    def mutate_def(self, mutateParams, parentTeam, teams, pActAtom, learner_id):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(ActionObject._actions)
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
                options = list(filter(lambda code: code != self.actionCode,mutateParams["actionCodes"]))
            else:
                options = mutateParams["actionCodes"]

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
                oldTeam = None
                if not self.isAtomic():
                    oldTeam = self.teamAction
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

    """
    Change action to team or atomic action.
    """
    def mutate_real(self, mutateParams, parentTeam, teams, pActAtom, learner_id):

        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(ActionObject._actions)
            self.teamAction = None
            print('0 valid_learners')
            return self

        # first maybe mutate just program
        if self.actionLength > 0 and flip(0.5):
            self.program.mutate(mutateParams)

        # mutate action
        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                options = list(filter(lambda code: code != self.actionCode, mutateParams["actionCodes"]))
            else:
                options = mutateParams["actionCodes"]

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = random.choice(options)
            self.actionLength = mutateParams["actionLengths"][self.actionCode]
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            # If we have a valid set of options choose from them
            if len(selection_pool) > 0:
                # let our current team know we won't be pointing to them anymore
                oldTeam = None
                if not self.isAtomic():
                    oldTeam = self.teamAction
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

class ConfActionObject1:

    def init_def(self, initParams:dict or int =None, action = None, _task='task'):

        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''
        from _tpg.team import Team1

        # The action is a team
        if isinstance(action, Team1):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
    

        # The action is another action object
        if isinstance(action, ActionObject1):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return

        # An int means the action is an index into the action codes in initParams
        if isinstance(action, int):
            if initParams is not None:
                if "actionCodes" not in initParams:
                    raise Exception('action codes not found in init params', initParams)

                try:
                    ActionObject1._actions = initParams["actionCodes"]
                    self.actionCode = initParams["actionCodes"][action]
                    self.teamAction = None
                except IndexError as err:
                    '''
                    TODO log index error
                    '''
                    print("Index error")
                return
            else:
                try:
                    self.actionCode=random.choice(ActionObject1._actions)
                    self.teamAction=None
                except:
                    print('諦めな・・・')
                return

    def init_real(self, initParams=None, action=None):

        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''
        from _tpg.team import Team1

        
        if isinstance(action, Team1):
            # The action is a team
            self.actionCode = None
            self.actionLength = None
            self.teamAction = action
            self.program = Program1(
                initParams=initParams, 
                maxProgramLength=initParams["initMaxActProgSize"],
                nOperations=initParams["nOperations"],
                nDestinations=initParams["nDestinations"],
                inputSize=initParams["inputSize"]
            )
        elif isinstance(action, ActionObject1):
            # The action is another action object
            self.actionCode = action.actionCode
            self.actionLength = action.actionLength
            self.teamAction = action.teamAction
            self.program = Program1(
                instructions=action.program.instructions,
                initParams=initParams
            )
        elif isinstance(action, int):
            # An int means the action is an index into the action codes in initParams
            if "actionCodes" not in initParams: raise Exception('action codes not found in init params', initParams)
            try:
                ActionObject1._actions = initParams["actionCodes"]
                self.actionCode = initParams["actionCodes"][action]
                self.actionLength = initParams["actionLengths"][action]
                self.teamAction = None
                self.program = Program1(
                    initParams=initParams, 
                    maxProgramLength=initParams["initMaxActProgSize"],
                    nOperations=initParams["nOperations"],
                    nDestinations=initParams["nDestinations"],
                    inputSize=initParams["inputSize"]
                )
            except IndexError as err:
                '''
                TODO log index error
                '''
                print("Index error")

        self.registers = np.zeros(max(initParams["nActRegisters"], initParams["nDestinations"]))

    """
    Returns the action code, and if applicable corresponding real action.
    """
    def getAction_def(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    """
    Returns the action code, and if applicable corresponding real action(s).
    """
    def getAction_real(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            if self.actionLength == 0:
                return self.actionCode, None
            else:
                return self.actionCode, self.getRealAction(state, actVars=actVars)

    """
    Gets the real action from a register.
    """
    def getRealAction_real(self, state, actVars=None):
        Program1.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[:self.actionLength]

    """
    Gets the real action from a register. With memory.
    """
    def getRealAction_real_mem(self, state, actVars=None):
        Program1.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        Program1.memWriteProbFunc)

        return self.registers[:self.actionLength]

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isAtomic_def(self):
        return self.teamAction is None

    """
    Change action to team or atomic action.
    """
    def mutate_def(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(ActionObject1._actions)
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
                options = list(filter(lambda code: code != self.actionCode,mutateParams["actionCodes"]))
            else:
                options = mutateParams["actionCodes"]

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
                oldTeam = None
                if not self.isAtomic():
                    oldTeam = self.teamAction
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

    """
    Change action to team or atomic action.
    """
    def mutate_real(self, mutateParams, parentTeam, teams, pActAtom, learner_id):

        # first maybe mutate just program
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(ActionObject1._actions)
            self.teamAction = None
            print('0 valid_learners')
            return self

        if self.actionLength > 0 and flip(0.5):
            self.program.mutate(mutateParams)

        # mutate action
        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                options = list(filter(lambda code: code != self.actionCode, mutateParams["actionCodes"]))
            else:
                options = mutateParams["actionCodes"]

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = random.choice(options)
            self.actionLength = mutateParams["actionLengths"][self.actionCode]
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            # If we have a valid set of options choose from them
            if len(selection_pool) > 0:
                # let our current team know we won't be pointing to them anymore
                oldTeam = None
                if not self.isAtomic():
                    oldTeam = self.teamAction
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

class ConfActionObject3:

    def init_def(self, initParams:dict or int =None, action = None, _task='task'):

        from _tpg.team import Team3

        # The action is a team
        if isinstance(action, Team3):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
    

        # The action is another action object
        if isinstance(action, ActionObject3):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return

        # An int means the action is an index into the action codes in initParams
        if isinstance(action, int):
            if initParams is not None:
                try:
                    # ActionObject3.actions=initParams["actionCodes"] #+ ActionObject3.nativeAction
                    self.actionCode = action
                    self.teamAction = None
                except IndexError as err:
                    '''
                    TODO log index error
                    '''
                    print("Index error")
                return
            else:
                try:
                    self.actionCode=random.randint(0, len(ActionObject3.actions)-1)
                    self.teamAction=None
                except:
                    print('諦めな・・・')
                return

    def init_real(self, initParams=None, action=None):

        from _tpg.team import Team3

        
        if isinstance(action, Team3):
            # The action is a team
            self.actionCode = None
            self.actionLength = None
            self.teamAction = action
            self.program = Program3(
                initParams=initParams, 
                maxProgramLength=initParams["initMaxActProgSize"],
                nOperations=initParams["nOperations"],
                nDestinations=initParams["nDestinations"],
                inputSize=initParams["inputSize"]
            )
        elif isinstance(action, ActionObject3):
            # The action is another action object
            self.actionCode = action.actionCode
            self.actionLength = action.actionLength
            self.teamAction = action.teamAction
            self.program = Program3(
                instructions=action.program.instructions,
                initParams=initParams
            )
        elif isinstance(action, int):
            # An int means the action is an index into the action codes in initParams
            if "actionCodes" not in initParams: raise Exception('action codes not found in init params', initParams)
            try:
                ActionObject3.actions = initParams["actionCodes"]
                self.actionCode = initParams["actionCodes"][action]
                self.actionLength = initParams["actionLengths"][action]
                self.teamAction = None
                self.program = Program3(
                    initParams=initParams, 
                    maxProgramLength=initParams["initMaxActProgSize"],
                    nOperations=initParams["nOperations"],
                    nDestinations=initParams["nDestinations"],
                    inputSize=initParams["inputSize"]
                )
            except IndexError as err:
                '''
                TODO log index error
                '''
                print("Index error")

        self.registers = np.zeros(max(initParams["nActRegisters"], initParams["nDestinations"]))

    """
    Returns the action code, and if applicable corresponding real action.
    """
    def getAction_def(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    """
    Returns the action code, and if applicable corresponding real action(s).
    """
    def getAction_real(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            if self.actionLength == 0:
                return self.actionCode, None
            else:
                return self.actionCode, self.getRealAction(state, actVars=actVars)

    """
    Gets the real action from a register.
    """
    def getRealAction_real(self, state, actVars=None):
        Program3.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[:self.actionLength]

    """
    Gets the real action from a register. With memory.
    """
    def getRealAction_real_mem(self, state, actVars=None):
        Program3.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        Program3.memWriteProbFunc)

        return self.registers[:self.actionLength]

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isAtomic_def(self):
        return self.teamAction is None

    """
    Change action to team or atomic action.
    """
    def mutate_def(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.randint(0,len(ActionObject3.actions)-1)
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
                options = list(filter(lambda code: code != self.actionCode,range(len(ActionObject3.actions))))
            else:
                options = range(len(ActionObject3.actions))

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = random.choice(options)
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams if t is not self.teamAction and t is not parentTeam]

            # If we have a valid set of options choose from them
            if len(selection_pool) > 0:

                # let our current team know we won't be pointing to them anymore
                if not self.isAtomic(): self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)

                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

        return self

    """
    Change action to team or atomic action.
    """
    def mutate_real(self, mutateParams, parentTeam, teams, pActAtom, learner_id):

        # first maybe mutate just program
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.randint(0, len(ActionObject3.actions)-1)
            self.teamAction = None
            print('0 valid_learners')
            return self

        if self.actionLength > 0 and flip(0.5):
            self.program.mutate(mutateParams)

        # mutate action
        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                options = list(filter(lambda code: code != self.actionCode, mutateParams["actionCodes"]))
            else:
                options = mutateParams["actionCodes"]

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = random.choice(options)
            self.actionLength = mutateParams["actionLengths"][self.actionCode]
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            # If we have a valid set of options choose from them
            if len(selection_pool) > 0:
                # let our current team know we won't be pointing to them anymore
                oldTeam = None
                if not self.isAtomic():
                    oldTeam = self.teamAction
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self
