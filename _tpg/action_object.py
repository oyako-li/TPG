# from logging import fatal

import numpy as np
import random
from _tpg.utils import flip

"""
Action  Object has a program to produce a value for the action, program doesn't
run if just a discrete action code.
"""
class _ActionObject:
    action=[0]
    Team = None

    # you should inherit
    def importance(self):
        from _tpg.team import _Team
        __class__.Team = _Team
    '''
    An action object can be initalized by:
        - Copying another action object
        - Passing an index into the action codes in initParams as the action
        - Passing a team as the action
    '''
    def __init__(self, initParams:dict or int =None, action = None, _task='task'):

        '''
        Defer importing the Team class to avoid circular dependency.
        This may require refactoring to fix properly
        '''
        self.importance(self)

        # The action is a team
        if isinstance(action, __class__.Team):
            self.teamAction = action
            self.actionCode = None
            #print("chose team action")
            return
    

        # The action is another action object
        if isinstance(action, __class__):
            self.actionCode = action.actionCode
            self.teamAction = action.teamAction
            return

        # An int means the action is an index into the action codes in initParams
        if isinstance(action, int):
            if initParams is not None:
                if "actionCodes" not in initParams:
                    raise Exception('action codes not found in init params', initParams)

                try:
                    __class__.action = initParams["actionCodes"]
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
                    self.actionCode=random.choice(__class__.action)
                    self.teamAction=None
                except:
                    print('諦めな・・・')
                return
    '''
    An ActionObject is equal to another object if that object:
        - is an instance of the ActionObject class
        - has the same action code
        - has the same team action
    '''
    def __eq__(self, __o:object)->bool:

        # The other object must be an instance of the ActionObject class
        if not isinstance(__o, __class__):    return False
        
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

    """
    Returns the action code, and if applicable corresponding real action(s).
    """
    def getAction(self, state, visited, actVars=None, path_trace=None):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited, actVars=actVars, path_trace=path_trace)
        else:
            # atomic action
            return self.actionCode

    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isAtomic(self):
        return self.teamAction is None


    """
    Change action to team or atomic action.
    """
    def mutate(self, mutateParams=None, parentTeam=None, teams=None, pActAtom=None, learner_id=None):
        # mutate action
        if any(item is None for item in (mutateParams, parentTeam, teams, pActAtom, learner_id)):
            self.actionCode = random.choice(__class__.action)
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
                if not self.isAtomic():
                    self.teamAction.inLearners.remove(str(learner_id))

                self.teamAction = random.choice(selection_pool)
                # Let the new team know we're pointing to them
                self.teamAction.inLearners.append(str(learner_id))

                #if oldTeam != None:
                #    print("Learner {} switched from Team {} to Team {}".format(learner_id, oldTeam.id, self.teamAction.id))
        
        return self
