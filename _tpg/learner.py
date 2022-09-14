from _tpg.utils import flip
import numpy as np
import collections
import uuid
import copy

class _Learner:
    Team = None
    ActionObject = None
    Program = None

    # you should inherit
    def importance(self):
        from _tpg.team import _Team
        from _tpg.action_object import _ActionObject
        from _tpg.program import _Program
        __class__.Team = _Team
        __class__.ActionObject = _ActionObject
        __class__.Program = _Program

    def __init__(self, 
        initParams:int or dict=0, 
        program=None, 
        actionObj=0, 
        numRegisters:int or np.ndarray=8, 
        _ancestor=None,
        _states:list=[],
        _inTeams:list=[],
        _frameNum:int=0
    ):
        self.importance()
        self.program = __class__.Program() if program is None else __class__.Program(instructions=program.instructions)
        self.actionObj = __class__.ActionObject(actionObj) if isinstance(actionObj, int) else __class__.ActionObject(action=actionObj,initParams=initParams)
        if isinstance(numRegisters, int): 
            self.registers = np.zeros(numRegisters, dtype=float) # 子供に記憶は継承されない。
        else: 
            self.registers = numRegisters
        if isinstance(initParams, int): 
            self.genCreate = initParams # Store the generation that this learner was created on
        elif isinstance(initParams, dict): 
            self.genCreate = initParams["generation"] # Store the generation that this learner was created on

        self.ancestor = _ancestor #By default no ancestor
        self.states = _states
        self.inTeams = _inTeams # Store a list of teams that reference this learner, incoming edges
        # self.actionCodes = initParams["actionCodes"]
        self.frameNum = _frameNum # Last seen frame is 0
        self.id = uuid.uuid4()


        if not self.isActionAtomic(): self.actionObj.teamAction.inLearners.append(str(self.id))


    """
    Get the bid value, highest gets its action selected.
    """
    def bid(self, state, actVars=None): 
        # exit early if we already got bidded this frame
        if self.frameNum == actVars["frameNum"]:
            return self.registers[0]

        self.frameNum = actVars["frameNum"]

        __class__.Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3],
                        actVars["memMatrix"], actVars["memMatrix"].shape[0], actVars["memMatrix"].shape[1],
                        __class__.Program.memWriteProb)

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

    def clone(self): 
        _clone = copy.deepcopy(self)
        _clone.inTeams = []
        _clone.id = uuid.uuid4()
        if _clone.actionObj.teamAction : _clone.actionObj.teamAction.inLearners.append(str(_clone.id))
        return _clone

    def zeroRegisters(self):
        self.registers = np.zeros(len(self.registers), dtype=float)
        self.actionObj.zeroRegisters()

    def numTeamsReferencing(self):
        return len(self.inTeams)


    def __eq__(self, __o: object) -> bool:
        # Object must be an instance of Learner
        if not isinstance(__o, __class__): return False

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
