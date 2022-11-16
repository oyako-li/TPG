from _tpg.utils import flip, _Logger
import numpy as np
import collections
import uuid
import copy

class _Learner(_Logger):
    Team = None
    ActionObject = None
    Program = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import _Team
            from _tpg.program import _Program
            from _tpg.action_object import _ActionObject
            cls.Team = _Team
            cls.Program = _Program
            cls.ActionObject = _ActionObject
           
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        program=None, 
        actionObj=None, 
        numRegisters:int or np.ndarray=8, 
        states:list=[],
        inTeams:list=[],
        frameNum:int=0,
        initParams:int or dict=0
    ):
        self.program = self.__class__.Program() if program is None else self.__class__.Program(instructions=program.instructions)
        self.actionObj = self.__class__.ActionObject(actionObj) if actionObj is None else self.__class__.ActionObject(action=actionObj,initParams=initParams)
        if isinstance(numRegisters, int): 
            self.registers = np.zeros(numRegisters, dtype=float) # 子供に記憶は継承されない。
        else: 
            self.registers = copy.deepcopy(numRegisters)
        if isinstance(initParams, int): 
            self.genCreate = initParams # Store the generation that this learner was created on
        elif isinstance(initParams, dict): 
            self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        # self.ancestor = _ancestor #By default no ancestor
        self.states = list(states)
        self.inTeams = list(inTeams) # Store a list of teams that reference this learner, incoming edges
        # self.actionCodes = initParams["actionCodes"]
        self.frameNum = frameNum # Last seen frame is 0
        self.id = uuid.uuid4()


        if not self.isActionAtomic(): self.actionObj.teamAction.inLearners.append(str(self.id))

    def __eq__(self, __o: object) -> bool:
        # Object must be an instance of Learner
        if not isinstance(__o, self.__class__): return False

        # The object must have been created the same generation as us
        if self.genCreate != __o.genCreate:   return False

        # The object's program must be equal to ours
        if self.program != __o.program:   return False

        # The object's action object must be equal to ours
        if self.actionObj != __o.actionObj:   return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(__o.inTeams): return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(__o.inTeams): return False

        # The other object's id must be equal to ours
        if self.id != __o.id: return False
        
        return True

    def __ne__(self, o:object)-> bool:
        '''
        Negation of __eq__
        '''
        return not self.__eq__(o)

    def __str__(self):
        '''
        String representation of a learner
        '''
        
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

    def bid(self, state, actVars=None): 
        """
        Get the bid value, highest gets its action selected.
        """
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]: return self.registers[0]

        self.frameNum = actVars["frameNum"]

        self.__class__.Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        self.__class__.Program.memWriteProb)

        return self.registers[0]

    def getAction(self, state, visited, actVars=None, path_trace=None): 
        """
        Returns the action of this learner, either atomic, or requests the action
        from the action team.
        """
        return self.actionObj.getAction(state, visited, actVars=actVars, path_trace=path_trace)

    def getActionTeam(self): 
        """
        Gets the team that is the action of the learners action object.
        """
        return self.actionObj.teamAction

    def isActionAtomic(self): 
        """
        Returns true if the action is atomic, otherwise the action is a team.
        """
        return self.actionObj.isAtomic()

    def mutate(self, mutateParams, parentTeam, teams, pActAtom): 
        """
        Mutates either the program or the action or both. 
        A mutation creates a new instance of the learner, removes it's anscestor and adds itself to the team.
        """
        
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

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.actionObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)

    @property
    def clone(self): 
        _clone = self.__class__(
            program = self.program,
            actionObj = self.actionObj,
            numRegisters=self.registers,
            states=self.states,
            inTeams=self.inTeams,
            frameNum=self.frameNum,
            initParams=self.genCreate
        )
        if not _clone.isActionAtomic(): 
            _clone.getActionTeam().inLearners.append(str(_clone.id))

        return _clone

class Learner1(_Learner):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team1
            from _tpg.action_object import _ActionObject
            from _tpg.program import Program1
            cls.Team = Team1
            cls.ActionObject = _ActionObject
            cls.Program = Program1
            
        return super().__new__(cls, *args, **kwargs)

class Learner1_1(Learner1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team1_1
            from _tpg.memory_object import ActionObject1
            from _tpg.program import Program1
            cls.Team = Team1_1
            cls.ActionObject = ActionObject1
            cls.Program = Program1
           
        return super().__new__(cls, *args, **kwargs)

class Learner1_2(Learner1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team1_2
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject2
            cls.Team = Team1_2
            cls.Program = Program1
            cls.ActionObject = ActionObject2

        return super().__new__(cls, *args, **kwargs)

class Learner1_3(Learner1):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_3
            from _tpg.program import Program1_3
            from _tpg.memory_object import Qualia
            
            cls._instance = True
            cls.Team = Team1_3
            cls.Program = Program1_3
            cls.ActionObject = Qualia
        return super().__new__(cls, *args, **kwargs)

    def bid(self, state, actVars=None): 
        """ Get the bid value, highest gets its action selected.

        Attribute:
            state: np.ndarray or Qualia
            actVars: hippocampus
                frameNum: randam timeout
                task: now task
                hippocampus: short term memory pool of Emulator
        
        Return:
            Qualia()[0]: number

        """
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.__class__.ActionObject(self.registers[0]).bid

        self.frameNum = actVars["frameNum"]

        self.__class__.Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars)

        return self.__class__.ActionObject(self.registers[0]).bid

class Learner1_3_1(Learner1_3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_3_1
            from _tpg.program import Program1_3
            from _tpg.memory_object import Qualia
            
            cls._instance = True
            cls.Team = Team1_3_1
            cls.Program = Program1_3
            cls.ActionObject = Qualia
        return super().__new__(cls, *args, **kwargs)

class Learner1_3_2(Learner1_3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_3_2
            from _tpg.program import Program1_3
            from _tpg.memory_object import Operation
            
            cls._instance = True
            cls.Team = Team1_3_2
            cls.Program = Program1_3
            cls.ActionObject = Operation
        return super().__new__(cls, *args, **kwargs)

class Learner2(_Learner):
    MemoryObject = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team2
            from _tpg.program import Program2
            from _tpg.memory_object import _MemoryObject
            cls.Team = Team2
            cls.Program = Program2
            cls.MemoryObject = _MemoryObject

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        program=None, 
        memoryObj=None, 
        numRegisters:int or np.ndarray=8, 
        states:list=[],
        inTeams:list=[],
        frameNum:int=0,
        initParams:int or dict=0
    ):
        self.program = self.__class__.Program() if program is None else self.__class__.Program(instructions=program.instructions)
        self.memoryObj = self.__class__.MemoryObject() if memoryObj is None else self.__class__.MemoryObject(state=memoryObj)
        if isinstance(numRegisters, int): 
            self.registers = np.zeros(numRegisters, dtype=float) # 子供に記憶は継承されない。
        else: 
            self.registers = copy.deepcopy(numRegisters)
        if isinstance(initParams, int): 
            self.genCreate = initParams # Store the generation that this learner was created on
        elif isinstance(initParams, dict): 
            self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        # self.ancestor = _ancestor #By default no ancestor
        self.states = list(states)
        self.inTeams = list(inTeams) # Store a list of teams that reference this learner, incoming edges
        # self.actionCodes = initParams["actionCodes"]
        self.frameNum = frameNum # Last seen frame is 0
        self.id = uuid.uuid4()

        if not self.isMemoryAtomic(): self.memoryObj.teamMemory.inLearners.append(str(self.id))
 
    def __eq__(self, __o: object) -> bool:
        # Object must be an instance of Learner
        if not isinstance(__o, self.__class__): return False

        # The object must have been created the same generation as us
        if self.genCreate != __o.genCreate:   return False

        # The object's program must be equal to ours
        if self.program != __o.program:   return False

        # The object's action object must be equal to ours
        if self.memoryObj != __o.memoryObj:   return False

        '''
        The other object's inTeams must match our own, therefore:
            - len(inTeams) must be equal
            - every id that appears in our inTeams must appear in theirs (order doesn't matter)
        '''
        if len(self.inTeams) != len(__o.inTeams): return False

        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inTeams) != collections.Counter(__o.inTeams): return False

        # The other object's id must be equal to ours
        if self.id != __o.id: return False
        
        return True

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.memoryObj.zeroRegisters()

    def mutate(self, mutateParams, parentTeam, teams, pMemAtom): 
        """
        Mutates either the program or the action or both. 
        A mutation creates a new instance of the learner, removes it's anscestor and adds itself to the team.
        """
        
        changed = False
        while not changed:
            # mutate the program
            if flip(mutateParams["pProgMut"]):
                changed = True
              
                self.program.mutate(mutateParams)

            # mutate the action
            if flip(mutateParams["pMemMut"]):

                changed = True
                
                self.memoryObj.mutate(mutateParams, parentTeam, teams, pMemAtom, learner_id=self.id)

        return self

    def isMemoryAtomic(self): 
        """
        Returns true if the action is atomic, otherwise the action is a team.
        """
        return self.memoryObj.isAtomic()

    def getMemoryTeam(self):
        """
        Gets the team that is the action of the learners action object.
        """
        return self.memoryObj.teamMemory

    def getImage(self, act, state, visited, memVars=None, path_trace=None): 
        """
        Returns the action of this learner, either atomic, or requests the action
        from the action team.
        """
        return self.memoryObj.getImage(act, state, visited, memVars=memVars, path_trace=path_trace)

    def bid(self, act, state, memVars=None): 
        """
        Get the bid value, highest gets its action selected.
        """
        # exit early if we already got bidded this frame
        if self.frameNum == memVars["frameNum"]:
            return self.registers[0]

        self.frameNum = memVars["frameNum"]

        self.__class__.Program.execute(act, state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        memVars["memMatrix"], memVars["memMatrix"].shape[0], memVars["memMatrix"].shape[1],
                        self.__class__.Program.memWriteProb)

        return self.registers[0]

    @property
    def clone(self): 
        _clone = self.__class__(
            program = self.program,
            memoryObj = self.memoryObj,
            numRegisters=self.registers,
            states=self.states,
            inTeams=self.inTeams,
            frameNum=self.frameNum,
            initParams=self.genCreate
        )
        if not _clone.isMemoryAtomic(): 
            _clone.getMemoryTeam().inLearners.append(str(_clone.id))

        return _clone

class Learner2_1(Learner2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team2_1
            from _tpg.program import Program2
            from _tpg.memory_object import MemoryObject
            cls.Team = Team2_1
            cls.Program = Program2
            cls.MemoryObject = MemoryObject

        return super().__new__(cls, *args, **kwargs)

class Learner2_2(Learner2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team2_2
            from _tpg.program import Program2_1
            from _tpg.memory_object import MemoryObject2
            cls.Team = Team2_2
            cls.Program = Program2_1
            cls.MemoryObject = MemoryObject2
           
        return super().__new__(cls, *args, **kwargs)

class Learner2_3(Learner2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.team import Team2_3
            from _tpg.program import Program2_3
            from _tpg.memory_object import Hippocampus
            cls.Team = Team2_3
            cls.Program = Program2_3
            cls.MemoryObject = Hippocampus
           
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        program=None, 
        memoryObj=None, 
        numRegisters:int or np.ndarray=8, 
        states:list=[],
        inTeams:list=[],
        frameNum:int=0,
        initParams:int or dict=0
    ):
        self.program = self.__class__.Program() if program is None else self.__class__.Program(instructions=program.instructions)
        self.memoryObj = self.__class__.MemoryObject() if memoryObj is None else self.__class__.MemoryObject(image=memoryObj)
        if isinstance(numRegisters, int): 
            self.registers = np.zeros(numRegisters, dtype=float) # 子供に記憶は継承されない。
        else: 
            self.registers = copy.deepcopy(numRegisters)
        if isinstance(initParams, int): 
            self.genCreate = initParams # Store the generation that this learner was created on
        elif isinstance(initParams, dict): 
            self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        # self.ancestor = _ancestor #By default no ancestor
        self.states = list(states)
        self.inTeams = list(inTeams) # Store a list of teams that reference this learner, incoming edges
        # self.actionCodes = initParams["actionCodes"]
        self.frameNum = frameNum # Last seen frame is 0
        self._id = uuid.uuid4()

        if not self.isMemoryAtomic(): self.memoryObj.teamMemory.inLearners.append(self.id)

    def getImage(self, state, visited, memVars=None, path_trace=None): 
        """
        Returns the action of this learner, either atomic, or requests the action
        from the action team.
        """
        return self.memoryObj.getImage(state, visited, memVars=memVars, path_trace=path_trace)

    def bid(self, state, memVars=None): 
        """
        Get the bid value, highest gets its action selected.
        """
        # exit early if we already got bidded this frame
        if self.frameNum == memVars["frameNum"]:
            return self.registers[0]

        self.frameNum = memVars["frameNum"]

        self.__class__.Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        memVars["memMatrix"], memVars["memMatrix"].shape[0], memVars["memMatrix"].shape[1],
                        self.__class__.Program.memWriteProb)

        return self.registers[0]

    @property
    def clone(self): 
        _clone = self.__class__(
            program = self.program,
            memoryObj = self.memoryObj,
            numRegisters=self.registers,
            states=self.states,
            inTeams=self.inTeams,
            frameNum=self.frameNum,
            initParams=self.genCreate
        )
        if not _clone.isMemoryAtomic(): 
            _clone.getMemoryTeam().inLearners.append(_clone.id)

        return _clone

    @property
    def id(self):
        return str(self._id)