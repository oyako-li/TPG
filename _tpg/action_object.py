# from logging import fatal

import numpy as np
import random
import pickle
from _tpg.utils import flip
from _tpg.memory_object import Memory1


"""
Action  Object has a program to produce a value for the action, program doesn't
run if just a discrete action code.
"""
class _ActionObject:
    actions=[0]
    Team = None
    _instance = None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import _Team
            cls._instance = True
            cls.Team = _Team
        return super().__new__(cls)
    '''
    An action object can be initalized by:
        - Copying another action object
        - Passing an index into the action codes in initParams as the action
        - Passing a team as the action
    '''
    def __init__(self,
        initParams:dict or int =None,
        action = None,
        _task='task'
    ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(action, self.__class__.Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return
        # An int means the action is an index into the action codes in initParams
        else:
            try:
                self.actionCode=random.choice(self.__class__.actions)
                self.teamAction=None
            except:
                print('諦めな・・・')
    '''
    An ActionObject is equal to another object if that object:
        - is an instance of the ActionObject class
        - has the same action code
        - has the same team action
    '''
    def __eq__(self, __o:object)->bool:

        # The other object must be an instance of the ActionObject class
        if not isinstance(__o, self.__class__):    return False
        
        # The other object's action code must be equal to ours
        if self.actionCode != __o.actionCode:     return False
        
        # The other object's team action must be equal to ours
        if self.teamAction != __o.teamAction:     return False

        return True

    '''
    Negate __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def __str__(self):
        return "TeamAction {} ActionCode: {}".format(
            self.teamAction if self.teamAction is not None else 'None',
            self.actionCode if self.actionCode is not None else 'None'
        )

    def zeroRegisters(self):
        try:
            self.registers = np.zeros(len(self.registers), dtype=float)
        except:
            pass

    def getAction(self, state, visited, actVars=None, path_trace=None):
        """
        Returns the action code, and if applicable corresponding real action(s).
        """
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    def isAtomic(self):
        """
        Returns true if the action is atomic, otherwise the action is a team.
        """
        return self.teamAction is None


    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        """
        Change action to team or atomic action.
        """
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(self.__class__.actions)
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
                options = list(filter(lambda code: code != self.actionCode,self.__class__.actions))
            else:
                options = self.__class__.actions

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
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self

class ActionObject1(_ActionObject):
    actions=Memory1()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.team import Team1_1
            cls._instance = True
            cls.Team = Team1_1

        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
        initParams:dict or int =None,
        action = None,
        _task='task'
    ):
        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''

        # The action is a team
        if isinstance(action, self.__class__.Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
        # The action is another action object
        elif isinstance(action, self.__class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return
        # An int means the action is an index into the action codes in initParams
        elif isinstance(action, str):
            self.actionCode = action
            self.teamAction = None
            return
        else:
            try:
                self.actionCode=self.__class__.actions.choice()
                self.teamAction=None
            except:
                print('諦めな・・・')

    def __getitem__(self, _key):
        return self.__class__.actions[self.actionCode][_key]

    def __add__(self, __o):
        return self.__class__.actions[self.actionCode]+__o

    def __sub__(self, __o):
        return self.__class__.actions[self.actionCode]-__o
    
    def getAction(self, _state, visited, actVars, path_trace=None):
        if self.teamAction is not None:
            return self.teamAction.act(_state, visited, actVars=actVars, path_trace=path_trace)
        else:
            assert self.actionCode in self.__class__.actions, f'{self.actionCode} is not in {self.__class__.actions}'
            self.__class__.actions.weights[self.actionCode]*=0.9 # 忘却確立減算
            self.__class__.actions.updateWeights()               # 忘却確立計上
            return self

    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = self.__class__.actions.choice([self.actionCode])
            self.teamAction = None
            return self

        if flip(pActAtom):
            # atomic
            '''
            If we already have an action code make sure not to pick the same one.
            TODO handle case where there is only 1 action code.
            '''
            if self.actionCode is not None:
                _ignore = self.actionCode
            else:
                _ignore = None

            # let our current team know we won't be pointing to them anymore
            if not self.isAtomic():
                #print("Learner {} switching from Team {} to atomic action".format(learner_id, self.teamAction.id))
                self.teamAction.inLearners.remove(str(learner_id))

            self.actionCode = self.__class__.actions.choice([_ignore])
            self.teamAction = None
        else:
            # team action
            selection_pool = [t for t in teams
                    if t is not self.teamAction and t is not parentTeam]

            if len(selection_pool) > 0:
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                self.teamAction.inLearners.append(str(learner_id))
       
        return self
 
    def backup(self, fileName):
        pickle.dump(self.__class__.actions, open(f'log/{fileName}-act.pickle', 'wb'))

    @classmethod
    def emulate(cls, fileName):
        _actions = pickle.load(open(fileName, 'rb'))
        cls.actions = _actions
        return cls.__init__()

    @property
    def action(self):
        return self.__class__.actions[self.actionCode]
