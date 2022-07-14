from _tpg.program import Program, Program1, Program2
from _tpg.action_object import ActionObject, ActionObject1, ActionObject2
from _tpg.utils import flip
import numpy as np
import collections
import uuid

class Learner:

    def __init__(self, initParams, program, actionObj, numRegisters, learner_id=None):
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
        
    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.actionObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)

    '''
    A learner is equal to another object if that object:
        - is an instance of the learner class
        - was created on the same generation
        - has an identical program
        - has the same action object
        - has the same inTeams
        - has the same id

    '''
    def __eq__(self, o: object) -> bool:
        
        # Object must be an instance of Learner
        if not isinstance(o, Learner):
            return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:
            return False

        # The object's program must be equal to ours
        if self.program != o.program:
            return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:
            return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams):
            return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams):
            return False

        # The other object's id must be equal to ours
        if self.id != o.id:
            return False
        
        return True

    def debugEq(self,o: object) -> bool:

        # Object must be an instance of Learner
        if not isinstance(o, Learner):
            print("other object is not instance of Learner")
            return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:
            print("other object has different genCreate")
            return False

        # The object's program must be equal to ours
        if self.program != o.program:
            print("other object has a different program")
            return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:
            print("other object has a different action object")
            return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams):
            print("other object has different number of inTeams")
            print("us:")
            print(self)
            print("other learner:")
            print(o)
            return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams):
            print("other object has different inTeams")
            return False

        # The other object's id must be equal to ours
        if self.id != o.id:
            print("other object has different id")
            return False
        
        return True

    '''
    Negation of __eq__
    '''
    def __ne__(self, o:object)-> bool:
        return not self.__eq__(o)

    '''
    String representation of a learner
    '''
    def __str__(self):
        
        result = """id: {}
created_at_gen: {}
program_id: {}
type: {}
action: {}
numTeamsReferencing: {}
inTeams:\n""".format(
                self.id,
                self.genCreate,
                self.program.id,
                "actionCode" if self.isActionAtomic() else "teamAction",
                self.actionObj.actionCode if self.isActionAtomic() else self.actionObj.teamAction.id,
                self.numTeamsReferencing()
            )

        for cursor in self.inTeams:
            result += "\t{}\n".format(cursor)
        
        return result

    """
    Create a new learner, either copied from the original or from a program or
    action. Either requires a learner, or a program/action pair.
    """


    """
    Get the bid value, highest gets its action selected.
    """
    def bid(self, state, actVars=None):
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[0]

    """
    Returns the action of this learner, either atomic, or requests the action
    from the action team.
    """
    def getAction(self, state, visited, actVars=None, path_trace=None):
        return self.actionObj.getAction(state, visited, actVars=actVars, path_trace=path_trace)

    """
    Gets the team that is the action of the learners action object.
    """
    def getActionTeam(self):
        return self.actionObj.teamAction

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isActionAtomic(self):
        return self.actionObj.isAtomic()


    """
    Mutates either the program or the action or both. 
    A mutation creates a new instance of the learner, removes it's anscestor and adds itself to the team.
    """
    def mutate(self, mutateParams, parentTeam, teams, pActAtom):


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

    def clone(self): pass
    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_learner import ConfLearner

        if functionsDict["init"] == "def":
            cls.__init__ = ConfLearner.init_def

        if functionsDict["bid"] == "def":
            cls.bid = ConfLearner.bid_def
        elif functionsDict["bid"] == "mem":
            cls.bid = ConfLearner.bid_mem

        if functionsDict["getAction"] == "def":
            cls.getAction = ConfLearner.getAction_def

        if functionsDict["getActionTeam"] == "def":
            cls.getActionTeam = ConfLearner.getActionTeam_def

        if functionsDict["isActionAtomic"] == "def":
            cls.isActionAtomic = ConfLearner.isActionAtomic_def

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfLearner.mutate_def
        if functionsDict["mutate"] == "def":
            cls.mutate = ConfLearner.mutate_def

class Learner1:

    def __init__(self, initParams, program, actionObj, numRegisters, learner_id=None)->None: pass

    """
    Get the bid value, highest gets its action selected.
    """
    def bid(self, state, actVars=None): pass

    """
    Returns the action of this learner, either atomic, or requests the action
    from the action team.
    """
    def getAction(self, state, visited, actVars=None, path_trace=None): pass

    """
    Gets the team that is the action of the learners action object.
    """
    def getActionTeam(self): pass

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isActionAtomic(self): pass

    """
    Mutates either the program or the action or both. 
    A mutation creates a new instance of the learner, removes it's anscestor and adds itself to the team.
    """
    def mutate(self, mutateParams, parentTeam, teams, pActAtom): pass

    def clone(self): pass

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.actionObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)

    def __eq__(self, o: object) -> bool:
        
        # Object must be an instance of Learner
        if not isinstance(o, Learner1): return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:   return False

        # The object's program must be equal to ours
        if self.program != o.program:   return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:   return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams): return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams): return False

        # The other object's id must be equal to ours
        if self.id != o.id: return False
        
        return True

    def debugEq(self,o: object) -> bool:

        # Object must be an instance of Learner
        if not isinstance(o, Learner1):
            print("other object is not instance of Learner")
            return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:
            print("other object has different genCreate")
            return False

        # The object's program must be equal to ours
        if self.program != o.program:
            print("other object has a different program")
            return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:
            print("other object has a different action object")
            return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams):
            print("other object has different number of inTeams")
            print("us:")
            print(self)
            print("other learner:")
            print(o)
            return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams):
            print("other object has different inTeams")
            return False

        # The other object's id must be equal to ours
        if self.id != o.id:
            print("other object has different id")
            return False
        
        return True

    '''
    Negation of __eq__
    '''
    def __ne__(self, o:object)-> bool:
        return not self.__eq__(o)

    '''
    String representation of a learner
    '''
    def __str__(self):
        
        result = """id: {}
                    created_at_gen: {}
                    program_id: {}
                    type: {}
                    action: {}
                    numTeamsReferencing: {}
                    inTeams:\n""".format(
                self.id,
                self.genCreate,
                self.program.id,
                "actionCode" if self.isActionAtomic() else "teamAction",
                self.actionObj.actionCode if self.isActionAtomic() else self.actionObj.teamAction.id,
                self.numTeamsReferencing()
            )

        for cursor in self.inTeams:
            result += "\t{}\n".format(cursor)
        
        return result
    
    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_learner import ConfLearner1

        if functionsDict["init"] == "def":
            cls.__init__ = ConfLearner1.init_def

        if functionsDict["bid"] == "def":
            cls.bid = ConfLearner1.bid_def
        elif functionsDict["bid"] == "mem":
            cls.bid = ConfLearner1.bid_mem

        if functionsDict["getAction"] == "def":
            cls.getAction = ConfLearner1.getAction_def

        if functionsDict["getActionTeam"] == "def":
            cls.getActionTeam = ConfLearner1.getActionTeam_def

        if functionsDict["isActionAtomic"] == "def":
            cls.isActionAtomic = ConfLearner1.isActionAtomic_def

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfLearner1.mutate_def

        if functionsDict["clone"] == "def":
            cls.clone = ConfLearner1.clone_def

class Learner2:

    def __init__(self, initParams, program, memoryObj, numRegisters, learner_id=None)->None: pass

    def bid(self, _act, _state, actVars=None): pass

    def getImage(self, _act, _state, visited, actVars=None, path_trace=None): pass

    def getMemoryTeam(self): pass

    def isMemoryAtomic(self): pass

    def mutate(self, mutateParams, parentTeam, teams, pMemAtom): pass

    def clone(self): pass

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.memoryObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)

    def __eq__(self, o: object) -> bool:
        
        # Object must be an instance of Learner
        if not isinstance(o, Learner2): return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:   return False

        # The object's program must be equal to ours
        if self.program != o.program:   return False

        # The object's action object must be equal to ours
        if self.memoryObj != o.memoryObj:   return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams): return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams): return False

        # The other object's id must be equal to ours
        if self.id != o.id: return False
        
        return True

    def debugEq(self,o: object) -> bool:

        # Object must be an instance of Learner
        if not isinstance(o, Learner2):
            print("other object is not instance of Learner")
            return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:
            print("other object has different genCreate")
            return False

        # The object's program must be equal to ours
        if self.program != o.program:
            print("other object has a different program")
            return False

        # The object's action object must be equal to ours
        if self.memoryObj != o.memoryObj:
            print("other object has a different action object")
            return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams):
            print("other object has different number of inTeams")
            print("us:")
            print(self)
            print("other learner:")
            print(o)
            return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams):
            print("other object has different inTeams")
            return False

        # The other object's id must be equal to ours
        if self.id != o.id:
            print("other object has different id")
            return False
        
        return True

    def __ne__(self, o:object)-> bool:
        return not self.__eq__(o)

    def __str__(self):
        
        result = """id: {}
                    created_at_gen: {}
                    program_id: {}
                    type: {}
                    memory: {}
                    numTeamsReferencing: {}
                    inTeams:\n""".format(
                self.id,
                self.genCreate,
                self.program.id,
                "memoryCode" if self.isMemoryAtomic() else "teamMemory",
                self.memoryObj.memoryCode if self.isMemoryAtomic() else self.memoryObj.teamMemory.id,
                self.numTeamsReferencing()
            )

        for cursor in self.inTeams:
            result += "\t{}\n".format(cursor)
        
        return result
    
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_learner import ConfLearner2

        if functionsDict["init"] == "def":
            cls.__init__ = ConfLearner2.init_def

        if functionsDict["bid"] == "def":
            cls.bid = ConfLearner2.bid_def
        elif functionsDict["bid"] == "mem":
            cls.bid = ConfLearner2.bid_mem

        if functionsDict["getImage"] == "def":
            cls.getImage = ConfLearner2.getImage_def

        if functionsDict["getMemoryTeam"] == "def":
            cls.getMemoryTeam = ConfLearner2.getMemoryTeam_def

        if functionsDict["isMemoryAtomic"] == "def":
            cls.isMemoryAtomic = ConfLearner2.isMemoryAtomic_def

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfLearner2.mutate_def

        if functionsDict["clone"] == "def":
            cls.clone = ConfLearner2.clone_def

class Learner3:

    def __init__(self, initParams, program, actionObj, numRegisters, learner_id=None)->None: pass

    """
    Get the bid value, highest gets its action selected.
    """
    def bid(self, state, actVars=None): pass

    """
    Returns the action of this learner, either atomic, or requests the action
    from the action team.
    """
    def getAction(self, state, visited, actVars=None, path_trace=None): pass

    """
    Gets the team that is the action of the learners action object.
    """
    def getActionTeam(self): pass

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isActionAtomic(self): pass

    """
    Mutates either the program or the action or both. 
    A mutation creates a new instance of the learner, removes it's anscestor and adds itself to the team.
    """
    def mutate(self, mutateParams, parentTeam, teams, pActAtom): pass

    def clone(self): pass

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.actionObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)

    def __eq__(self, o: object) -> bool:
        
        # Object must be an instance of Learner
        if not isinstance(o, Learner3): return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:   return False

        # The object's program must be equal to ours
        if self.program != o.program:   return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:   return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams): return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams): return False

        # The other object's id must be equal to ours
        if self.id != o.id: return False
        
        return True

    def debugEq(self,o: object) -> bool:

        # Object must be an instance of Learner
        if not isinstance(o, Learner3):
            print("other object is not instance of Learner")
            return False

        # The object must have been created the same generation as us
        if self.genCreate != o.genCreate:
            print("other object has different genCreate")
            return False

        # The object's program must be equal to ours
        if self.program != o.program:
            print("other object has a different program")
            return False

        # The object's action object must be equal to ours
        if self.actionObj != o.actionObj:
            print("other object has a different action object")
            return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(o.inTeams):
            print("other object has different number of inTeams")
            print("us:")
            print(self)
            print("other learner:")
            print(o)
            return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(o.inTeams):
            print("other object has different inTeams")
            return False

        # The other object's id must be equal to ours
        if self.id != o.id:
            print("other object has different id")
            return False
        
        return True

    '''
    Negation of __eq__
    '''
    def __ne__(self, o:object)-> bool:
        return not self.__eq__(o)

    '''
    String representation of a learner
    '''
    def __str__(self):
        
        result = """id: {}
                    created_at_gen: {}
                    program_id: {}
                    type: {}
                    action: {}
                    numTeamsReferencing: {}
                    inTeams:\n""".format(
                self.id,
                self.genCreate,
                self.program.id,
                "actionCode" if self.isActionAtomic() else "teamAction",
                self.actionObj.actionCode if self.isActionAtomic() else self.actionObj.teamAction.id,
                self.numTeamsReferencing()
            )

        for cursor in self.inTeams:
            result += "\t{}\n".format(cursor)
        
        return result
    
    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_learner import ConfLearner3

        if functionsDict["init"] == "def":
            cls.__init__ = ConfLearner3.init_def

        if functionsDict["bid"] == "def":
            cls.bid = ConfLearner3.bid_def
        elif functionsDict["bid"] == "mem":
            cls.bid = ConfLearner3.bid_mem

        if functionsDict["getAction"] == "def":
            cls.getAction = ConfLearner3.getAction_def

        if functionsDict["getActionTeam"] == "def":
            cls.getActionTeam = ConfLearner3.getActionTeam_def

        if functionsDict["isActionAtomic"] == "def":
            cls.isActionAtomic = ConfLearner3.isActionAtomic_def

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfLearner3.mutate_def

        if functionsDict["clone"] == "def":
            cls.clone = ConfLearner3.clone_def

