import uuid
from _tpg.utils import flip, breakpoint, abstract, _Logger
import random
import collections
import numpy as np

"""
The main building block of TPG. Each team has multiple learning which decide the
action to take in the graph.
"""

class _Team(_Logger):
    Learner = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import _Learner
            cls._instance = True
            cls.Learner = _Learner
        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
            learners:list = [],
            inLearners:list = [],
            outcomes:dict = {'task':0.},
            fitness:float = 0.0,
            extinction:float = 1.0,
            initParams:int or dict=0
        ): 
        self.learners = list(learners)
        self.inLearners = list(inLearners) # ids of learners referencing this team
        self.outcomes =  dict(outcomes)# scores at various tasks
        self.id = uuid.uuid4()
        self.fitness = fitness
        self.sequence = []
        self.extinction=extinction
        if isinstance(initParams, dict): self.genCreate = initParams["generation"]
        elif isinstance(initParams, int): self.genCreate = initParams

    def __str__(self) -> str:
        return f"""
        learners:{self.learners},
        inLearners:{self.inLearners},
        outcomes:{self.outcomes},
        id:{self.id},
        fitness:{self.fitness},
        sequence:{self.sequence},
        extinction:{self.extinction}
        """

    def __eq__(self, __o: object) -> bool: 
        # Object must be instance of Team
        if not isinstance(__o, self.__class__):    return False

        # Object must be created the same generation as us
        if self.genCreate != __o.genCreate:   return False
        
        '''
        The other object's learners must match our own, therefore:
            - len(self.learners) must be equal to len(o.learners)
            - every learner that appears in our list of learners must appear in the 
              other object's list of learners as well.
        '''
        if len(self.learners) != len(__o.learners):   return False
        
        for l in self.learners:
            if l not in __o.learners:
                return False

        
        '''
        The other object's inLearners must match our own, therefore:
            - len(self.inLearners) must be equal to len(o.inLearners)
            - every learner that appears in our list of inLearners must appear in 
              the other object's list of inLearners as well. 
        '''
        if len(self.inLearners) != len(__o.inLearners):   return False
        
        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inLearners) != collections.Counter(__o.inLearners):   return False

        # The other object's id must be equal to ours
        if self.id != __o.id: return False

        return True

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __getitem__(self, _task):
        # assert _task in self.outcomes.keys(), f'{_task} is not in {self.outcomes.keys()}'
        if not self.outcomes.get(_task): self.outcomes[_task]=0.
        return self.outcomes[_task]
    
    def __setitem__(self, _task, _score):
        self.outcomes[_task]=_score

    def __hash__(self):
        return int(self.id)

    def _mutation_delete(self, probability):

            original_probability = float(probability)

            if probability == 0.0:
                return []

            if probability >= 1.0: 
                raise Exception("pLrnDel is greater than or equal to 1.0!")

            # Freak out if we don't have an atomic action
            if self.numAtomicActions() < 1: 
                raise Exception("Less than one atomic action in team! This shouldn't happen", self)


            deleted_learners = []

            # delete some learners
            while flip(probability) and len(self.learners) > 2: # must have >= 2 learners
                probability *= original_probability # decrease next chance


                # If we have more than one learner with an atomic action pick any learner to delete
                if self.numAtomicActions() > 1:
                    learner = random.choice(self.learners)
                else: 
                    # Otherwise if we only have one, filter it out and pick from the remaining learners
                    valid_choices = list(filter(lambda x: not x.isActionAtomic(), self.learners)) # isActionAtomic以外から削除を決定。
                    learner = random.choice(valid_choices)

                deleted_learners.append(learner)
                self.removeLearner(learner)

            return deleted_learners

    def _mutation_add(self, probability, maxTeamSize, selection_pool):

        original_probability = float(probability)

        # Zero chance to add anything, return right away
        if probability == 0.0 or len(selection_pool) == 0 or (maxTeamSize > 0 and len(self.learners) >= maxTeamSize):   return []
        
        # If this were true, we'd end up adding the entire selection pool
        if probability >= 1.0:  raise Exception("pLrnAdd is greater than or equal to 1.0!")

        added_learners = []  
        while flip(probability) and (maxTeamSize <= 0 or len(self.learners) < maxTeamSize):
            # If no valid selections left, break out of the loop
            if len(selection_pool) == 0:    break

            probability *= original_probability # decrease next chance


            learner = random.choice(selection_pool)
            added_learners.append(learner)
            self.addLearner(learner)

            # Ensure we don't pick the same learner twice by filtering the learners we've added from the selection pool
            selection_pool = list(filter(lambda x:x not in added_learners, selection_pool))

        return added_learners

    def _mutation_mutate(self, probability, mutateParams, teams):
        mutated_learners = {}
        '''
         This original learners thing is important, otherwise may mutate learners that we just added through mutation. 
         This breaks reference tracking because it results in 'ghost learners' that were created during mutation, added themselves 
         to inLearners in the teams they pointed to, but them were mutated out before being tracked by the trainer. So you end up
         with teams hold a record in their inLearners to a learner that doesn't exist
        '''
        original_learners = list(self.learners)
        new_learners = []
        for learner in original_learners:
            if flip(probability):

                # If we only have one learner with an atomic action and the current learner is it
                if self.numAtomicActions() == 1 and learner.isActionAtomic():
                    pActAtom0 = 1.1 # Ensure their action remains atomic
                else:
                    # Otherwise let there be a probability that the learner's action is atomic as defined in the mutate params
                    pActAtom0 = mutateParams['pActAtom']

                #print("Team {} creating learner".format(self.id))
                # Create a new learner 
                newLearner = self.__class__.Learner(
                    program=learner.program, 
                    actionObj=learner.actionObj, 
                    numRegisters=len(learner.registers), 
                    initParams=mutateParams,
                    frameNum=learner.frameNum
                )
                new_learners.append(newLearner)
                # Add the mutated learner to our learners
                # Must add before mutate so that the new learner has this team in its inTeams
                self.addLearner(newLearner)


                # mutate it
                newLearner.mutate(mutateParams, self, teams, pActAtom0)
                # Remove the existing learner from the team
                self.removeLearner(learner)

                mutated_learners[str(learner.id)] = str(newLearner.id)

      
        return mutated_learners, new_learners

    def act(self, state, visited, actVars=None, path_trace=None): 
        # If we've already visited me, throw an exception
        assert not str(self.id) in visited, f"Already visited team {str(self.id)}"

        # Add this team's id to the list of visited ids
        visited.append(str(self.id)) 
        if len(self.learners)==0:
            print('0 valid')
            self.addLearner(self.__class__.Learner())

        '''
        Valid learners are ones which:
            * Are action atomic
            * Whose team we have not yet visited
        '''
        valid_learners = [lrnr for lrnr in self.learners if lrnr.isActionAtomic() or str(lrnr.getActionTeam().id) not in visited]
        
        if len(valid_learners)==0: 

            mutate_learner = random.choice(self.learners)
            clone = mutate_learner.clone
            if not clone.isActionAtomic(): clone.actionObj.teamAction.inLearner.remove(str(clone.id))
            clone.actionObj.mutate()

            self.addLearner(clone)
            valid_learners.append(clone)

        top_learner = max(valid_learners, key=lambda lrnr: lrnr.bid(state, actVars=actVars))
    
        # If we're tracing this path
        if path_trace != None:
            
            last_segment = path_trace[-1] if len(path_trace) != 0 else None

            # Create our path segment
            path_segment =  {
                'team_id': str(self.id),
                'top_learner': str(top_learner.id),
                'top_bid': top_learner.bid(state, actVars=actVars),
                'top_action': top_learner.actionObj.actionCode if top_learner.isActionAtomic() else str(top_learner.actionObj.teamAction.id),
                'depth': last_segment['depth'] + 1 if last_segment != None else 0,# Record path depth
                'bids': []
            }

            # Populate bid values
            for cursor in valid_learners:
                path_segment['bids'].append({
                    'learner_id': str(cursor.id),
                    'bid': cursor.bid(state, actVars=actVars),
                    'action': cursor.actionObj.actionCode if cursor.isActionAtomic() else str(cursor.actionObj.teamAction.id)
                })

            # Append our path segment to the trace
            path_trace.append(path_segment)

        return top_learner.getAction(state, visited=visited, actVars=actVars, path_trace=path_trace)

    def addLearner(self, learner): 
        self.learners.append(learner)
        learner.inTeams.append(str(self.id)) # Add this team's id to the list of teams that reference the learner

        return True

    def removeLearner(self, learner): 
        # only delete if actually in this team
        '''
        TODO log the attempt to remove a learner that doesn't appear in this team
        '''
        if learner not in self.learners:
            raise Exception("Attempted to remove a learner ({}) not referenced by team {}".format(
            str(learner.id), str(self.id)
        ))

        # Find the learner to remove
        to_remove = [cursor for  cursor in self.learners if cursor == learner]
        if len(to_remove) != 1:
            raise Exception("Duplicate learner detected during team.removeLearner. {} duplicates".format(len(to_remove)))
        to_remove = to_remove[0]

        # Build a new list of learners containing only learners that are not the learner
        self.learners = [cursor for cursor in self.learners if cursor != learner ]

        # Remove our id from the learner's inTeams
        # NOTE: Have to do this after removing the learner otherwise, removal will fail 
        # since the learner's inTeams will not match 
        to_remove.inTeams.remove(str(self.id))

    def removeLearners(self): 
        for learner in self.learners:
            learner.inTeams.remove(str(self.id))

        del self.learners[:]

    def numAtomicActions(self):
        num = 0
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                num += 1

        return num

    def mutate(self, mutateParams, allLearners, teams):
        '''
        With rampant mutations every mutateParams["rampantGen"] generations we do X extra
        iterations of mutation. Where X is a random number between mutateParams["rampantMin"] 
        and mutateParams["rampantMax"].
        '''
        # Throw an error if rampantMin is undefined but 

        # Throw an error if rampantMin > rampant Max
        if mutateParams['rampantGen'] != 0 and mutateParams['rampantMin'] > mutateParams['rampantMax']:
            raise Exception("Min rampant iterations is greater than max rampant iterations!", mutateParams)
        
        if (mutateParams["rampantGen"] > 0 and # The rapantGen cannot be 0, as x mod 0 is undefined
            mutateParams["generation"] % mutateParams["rampantGen"] == 0 and # Determine if this is a rampant generation
            mutateParams["generation"] > mutateParams["rampantGen"]  # Rampant generations cannot occur before generation passes rampantGen
            ): 
            rampantReps = random.randrange(mutateParams["rampantMin"], mutateParams["rampantMax"]) if mutateParams['rampantMin'] < mutateParams['rampantMax'] else mutateParams['rampantMin']
        else:
            rampantReps = 1

        # increase diversity by repeating mutations

        mutation_delta = {}
        new_learners = []

        for i in range(rampantReps):
            #print("i/rampant reps:  {}/{} ".format(i, rampantReps))
            # delete some learners
            '''
            TODO log mutation deltas...
            '''
            deleted_learners = self._mutation_delete(mutateParams["pLrnDel"])

            # Create a selection pool from which to add learners to this team
            
            # Filter out learners that already belong to this team
            selection_pool = list(filter(lambda x: x not in self.learners, allLearners))
            
            # Filter out learners that point to this team
            selection_pool = list(filter(lambda x: str(x.id) not in self.inLearners, selection_pool))

            # Filter out learners we just deleted
            selection_pool = list(filter(lambda x: x not in deleted_learners, selection_pool))
            
            added_learners = self._mutation_add(mutateParams["pLrnAdd"], mutateParams["maxTeamSize"], selection_pool)

            # give chance to mutate all learners
            mutated_learners, mutation_added_learners = self._mutation_mutate(mutateParams["pLrnMut"], mutateParams, teams)
            new_learners += mutation_added_learners

            # Compile mutation_delta for this iteration
            mutation_delta[i] = {} 
            mutation_delta[i]['deleted_learners'] = deleted_learners
            mutation_delta[i]['added_learners'] = added_learners
            mutation_delta[i]['mutated_learners'] = mutated_learners

        for cursor in new_learners:
            if cursor in self.learners:
                new_learners.remove(cursor)

        for cursor in new_learners:
            if len(cursor.inTeams) == 0 and not cursor.isActionAtomic():
                cursor.actionObj.teamAction.inLearners.remove(str(cursor.id))

        # return the number of iterations of mutation
        return rampantReps, mutation_delta, new_learners

    def zeroRegisters(self):
        for learner in self.learners:
            learner.zeroRegisters()

    def numLearnersReferencing(self):
        return len(self.inLearners)

    @property
    def clone(self): 
        _clone = self.__class__(
            outcomes=self.outcomes,
            fitness=self.fitness,
            extinction=self.extinction,
        )
        for learner in self.learners:
            _clone.addLearner(learner.clone)
        
        assert _clone.id != self.id, f'clone cant make'

        return _clone

class Team1(_Team):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1

            cls._instance = True
            cls.Learner = Learner1

        return super().__new__(cls, *args, **kwargs)

class Team1_1(Team1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_1
            cls._instance = True
            cls.Learner = Learner1_1

        return super().__new__(cls, *args, **kwargs)

class Team1_2(Team1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_2
            cls._instance = True
            cls.Learner = Learner1_2

        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
            learners:list = [],
            inLearners:list = [],
            outcomes:dict = {'task':0.},
            fitness:float = 0.0,
            extinction:float = 1.0,
            initParams:int or dict=0
        ): 
        self.learners = list(learners)
        self.inLearners = list(inLearners) # ids of learners referencing this team
        self.outcomes =  dict(outcomes)# scores at various tasks
        self._id = uuid.uuid4()
        self.fitness = fitness
        self.sequence = list()
        self.extinction=extinction
        if isinstance(initParams, dict): self.genCreate = initParams["generation"]
        elif isinstance(initParams, int): self.genCreate = initParams

    def __eq__(self, __o: object) -> bool: 
        # Object must be instance of Team
        if not isinstance(__o, self.__class__):    return False

        # Object must be created the same generation as us
        if self.genCreate != __o.genCreate:   return False
        
        '''
        The other object's learners must match our own, therefore:
            - len(self.learners) must be equal to len(o.learners)
            - every learner that appears in our list of learners must appear in the 
              other object's list of learners as well.
        '''
        if len(self.learners) != len(__o.learners):   return False
        
        for l in self.learners:
            if l not in __o.learners:
                return False

        
        '''
        The other object's inLearners must match our own, therefore:
            - len(self.inLearners) must be equal to len(o.inLearners)
            - every learner that appears in our list of inLearners must appear in 
              the other object's list of inLearners as well. 
        '''
        if len(self.inLearners) != len(__o.inLearners):   return False
        
        '''
        Collection comparison via collection counters
        https://www.journaldev.com/37089/how-to-compare-two-lists-in-python
        '''
        if collections.Counter(self.inLearners) != collections.Counter(__o.inLearners):   return False

        # The other object's id must be equal to ours
        if self._id != __o._id: return False

        return True

    def _mutation_delete(self, probability):

            original_probability = float(probability)

            if probability == 0.0:
                return []

            if probability >= 1.0: 
                raise Exception("pLrnDel is greater than or equal to 1.0!")

            # Freak out if we don't have an atomic action
            if self.numAtomicActions() < 1: 
                raise Exception("Less than one atomic action in team! This shouldn't happen", self)


            deleted_learners = []

            # delete some learners
            while flip(probability) and len(self.learners) > 2: # must have >= 2 learners
                probability *= original_probability # decrease next chance


                # If we have more than one learner with an atomic action pick any learner to delete
                if self.numAtomicActions() > 1:
                    learner = random.choice(self.learners)
                else: 
                    # Otherwise if we only have one, filter it out and pick from the remaining learners
                    valid_choices = list(filter(lambda x: not x.isActionAtomic(), self.learners)) # isActionAtomic以外から削除を決定。
                    learner = random.choice(valid_choices)

                deleted_learners.append(learner)
                self.removeLearner(learner)

            return deleted_learners

    def _mutation_add(self, probability, maxTeamSize, selection_pool):

        original_probability = float(probability)

        # Zero chance to add anything, return right away
        if probability == 0.0 or len(selection_pool) == 0 or (maxTeamSize > 0 and len(self.learners) >= maxTeamSize):   return []
        
        # If this were true, we'd end up adding the entire selection pool
        if probability >= 1.0:  raise Exception("pLrnAdd is greater than or equal to 1.0!")

        added_learners = []  
        while flip(probability) and (maxTeamSize <= 0 or len(self.learners) < maxTeamSize):
            # If no valid selections left, break out of the loop
            if len(selection_pool) == 0:    break

            probability *= original_probability # decrease next chance


            learner = random.choice(selection_pool)
            added_learners.append(learner)
            self.addLearner(learner)

            # Ensure we don't pick the same learner twice by filtering the learners we've added from the selection pool
            selection_pool = list(filter(lambda x:x not in added_learners, selection_pool))

        return added_learners

    def _mutation_mutate(self, probability, mutateParams, teams):
        mutated_learners = {}
        '''
         This original learners thing is important, otherwise may mutate learners that we just added through mutation. 
         This breaks reference tracking because it results in 'ghost learners' that were created during mutation, added themselves 
         to inLearners in the teams they pointed to, but them were mutated out before being tracked by the trainer. So you end up
         with teams hold a record in their inLearners to a learner that doesn't exist
        '''
        original_learners = list(self.learners)
        new_learners = []
        for learner in original_learners:
            if flip(probability):

                # If we only have one learner with an atomic action and the current learner is it
                if self.numAtomicActions() == 1 and learner.isActionAtomic():
                    pActAtom0 = 1.1 # Ensure their action remains atomic
                else:
                    # Otherwise let there be a probability that the learner's action is atomic as defined in the mutate params
                    pActAtom0 = mutateParams['pActAtom']

                #print("Team {} creating learner".format(self.id))
                # Create a new learner 
                newLearner = self.__class__.Learner(
                    program=learner.program, 
                    actionObj=learner.actionObj, 
                    numRegisters=len(learner.registers), 
                    initParams=mutateParams,
                    frameNum=learner.frameNum
                )
                new_learners.append(newLearner)
                # Add the mutated learner to our learners
                # Must add before mutate so that the new learner has this team in its inTeams
                self.addLearner(newLearner)


                # mutate it
                newLearner.mutate(mutateParams, self, teams, pActAtom0)
                # Remove the existing learner from the team
                self.removeLearner(learner)

                mutated_learners[learner._id] = newLearner._id

      
        return mutated_learners, new_learners

    def act(self, state, visited, actVars=None, path_trace=None): 
        # If we've already visited me, throw an exception
        assert not self._id in visited, f"Already visited team {self._id}"

        self.extinction*=0.9

        # Add this team's_id to the list of visited_ids
        visited.append(self._id) 
        if len(self.learners)==0:
            print('0 valid')
            self.addLearner(self.__class__.Learner())

        '''
        Valid learners are ones which:
            * Are action atomic
            * Whose team we have not yet visited
        '''
        valid_learners = [lrnr for lrnr in self.learners if lrnr.isActionAtomic() or lrnr.getActionTeam()._id not in visited]
        
        if len(valid_learners)==0: 

            mutate_learner = random.choice(self.learners)
            clone = mutate_learner.clone
            if not clone.isActionAtomic(): clone.actionObj.teamAction.inLearner.remove(clone._id)
            clone.actionObj.mutate()

            self.addLearner(clone)
            valid_learners.append(clone)

        top_learner = max(valid_learners, key=lambda lrnr: lrnr.bid(state, actVars=actVars))
    
        # If we're tracing this path
        if path_trace != None:
            
            last_segment = path_trace[-1] if len(path_trace) != 0 else None

            # Create our path segment
            path_segment =  {
                'team_id': self.id,
                'top_learner': top_learner.id,
                'top_bid': top_learner.bid(state, actVars=actVars),
                'top_action': top_learner.actionObj.actionCode if top_learner.isActionAtomic() else top_learner.actionObj.teamAction._id,
                'depth': last_segment['depth'] + 1 if last_segment != None else 0,# Record path depth
                'b_ids': []
            }

            # Populate bid values
            for cursor in valid_learners:
                path_segment['bids'].append({
                    'learner_id': cursor.id,
                    'bid': cursor.bid(state, actVars=actVars),
                    'action': cursor.actionObj.actionCode if cursor.isActionAtomic() else cursor.actionObj.teamAction._id
                })

            # Append our path segment to the trace
            path_trace.append(path_segment)

        return top_learner.getAction(state, visited=visited, actVars=actVars, path_trace=path_trace)

    def addLearner(self, learner): 
        self.learners.append(learner)
        learner.inTeams.append(self._id) # Add this team's_id to the list of teams that reference the learner

        return True

    def appendSequence(self, _sequence):
        if len(_sequence)<1: return
        self.sequence=np.array(_sequence)
        # self.addSequence()

    def addSequence(self):
        # self.debug(f'append_sequence:{self.sequence}')
        if len(self.sequence)<1: return
        # assert self.sequence != [], f'{self.sequence}'
        # sequence = abstract(self.sequence)
        sequence_learner = self.__class__.Learner(actionObj=self.sequence)
        self.addLearner(sequence_learner)
        # self.debug(f'appended_sequence:{sequence_learner.actionObj}')

    def removeLearner(self, learner): 
        # only delete if actually in this team
        '''
        TODO log the attempt to remove a learner that doesn't appear in this team
        '''
        if learner not in self.learners:
            raise Exception("Attempted to remove a learner ({}) not referenced by team {}".format(
            learner._id, self._id
        ))

        # Find the learner to remove
        to_remove = [cursor for  cursor in self.learners if cursor == learner]
        if len(to_remove) != 1:
            raise Exception("Duplicate learner detected during team.removeLearner. {} duplicates".format(len(to_remove)))
        to_remove = to_remove[0]

        # Build a new list of learners containing only learners that are not the learner
        self.learners = [cursor for cursor in self.learners if cursor != learner ]

        # Remove our_id from the learner's inTeams
        # NOTE: Have to do this after removing the learner otherwise, removal will fail 
        # since the learner's inTeams will not match 
        to_remove.inTeams.remove(self._id)

    def removeLearners(self): 
        for learner in self.learners:
            learner.inTeams.remove(self._id)

        del self.learners[:]

    def mutate(self, mutateParams, allLearners, teams):
        '''
        With rampant mutations every mutateParams["rampantGen"] generations we do X extra
        iterations of mutation. Where X is a random number between mutateParams["rampantMin"] 
        and mutateParams["rampantMax"].
        '''
        # Throw an error if rampantMin is undefined but 

        # Throw an error if rampantMin > rampant Max
        if mutateParams['rampantGen'] != 0 and mutateParams['rampantMin'] > mutateParams['rampantMax']:
            raise Exception("Min rampant iterations is greater than max rampant iterations!", mutateParams)
        
        if (mutateParams["rampantGen"] > 0 and # The rapantGen cannot be 0, as x mod 0 is undefined
            mutateParams["generation"] % mutateParams["rampantGen"] == 0 and # Determine if this is a rampant generation
            mutateParams["generation"] > mutateParams["rampantGen"]  # Rampant generations cannot occur before generation passes rampantGen
            ): 
            rampantReps = random.randrange(mutateParams["rampantMin"], mutateParams["rampantMax"]) if mutateParams['rampantMin'] < mutateParams['rampantMax'] else mutateParams['rampantMin']
        else:
            rampantReps = 1

        # increase diversity by repeating mutations

        mutation_delta = {}
        new_learners = []

        for i in range(rampantReps):
            #print("i/rampant reps:  {}/{} ".format(i, rampantReps))
            # delete some learners
            '''
            TODO log mutation deltas...
            '''
            deleted_learners = self._mutation_delete(mutateParams["pLrnDel"])

            # Create a selection pool from which to add learners to this team
            
            # Filter out learners that already belong to this team
            selection_pool = list(filter(lambda x: x not in self.learners, allLearners))
            
            # Filter out learners that point to this team
            selection_pool = list(filter(lambda x: x._id not in self.inLearners, selection_pool))
            

            # Filter out learners we just deleted
            selection_pool = list(filter(lambda x: x not in deleted_learners, selection_pool))
            
            added_learners = self._mutation_add(mutateParams["pLrnAdd"], mutateParams["maxTeamSize"], selection_pool)

            # give chance to mutate all learners
            mutated_learners, mutation_added_learners = self._mutation_mutate(mutateParams["pLrnMut"], mutateParams, teams)
            new_learners += mutation_added_learners

            # Compile mutation_delta for this iteration
            mutation_delta[i] = {} 
            mutation_delta[i]['deleted_learners'] = deleted_learners
            mutation_delta[i]['added_learners'] = added_learners
            mutation_delta[i]['mutated_learners'] = mutated_learners

        for cursor in new_learners:
            if cursor in self.learners:
                new_learners.remove(cursor)

        for cursor in new_learners:
            if len(cursor.inTeams) == 0 and not cursor.isActionAtomic():
                cursor.actionObj.teamAction.inLearners.remove(cursor._id)

        # return the number of iterations of mutation
        return rampantReps, mutation_delta, new_learners

    def numAtomicActions(self):
        num = 0
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                num += 1

        return num

    def zeroRegisters(self):
        for learner in self.learners:
            learner.zeroRegisters()

    def numLearnersReferencing(self):
        return len(self.inLearners)

    @property
    def id(self):
        return str(self._id)

class Team1_2_1(Team1_2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_2_1
            cls._instance = True
            cls.Learner = Learner1_2_1

        return super().__new__(cls, *args, **kwargs)

class Team1_2_2(Team1_2_1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_2_2
            cls._instance = True
            cls.Learner = Learner1_2_2

        return super().__new__(cls, *args, **kwargs)

class Team1_2_3(Team1_2_2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_2_3
            cls._instance = True
            cls.Learner = Learner1_2_3

        return super().__new__(cls, *args, **kwargs)
     
class Team1_3(Team1):
    """ birth child to rootteam with preference
    
    """

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_3
            cls._instance = True
            cls.Learner = Learner1_3
        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
            learners:list = [],
            inLearners:list = [],
            outcomes:dict = {'task':0.},
            fitness:float = 0.0,
            initParams:int or dict=0
        ): 
        self.learners = list(learners)
        self.inLearners = list(inLearners)  # ids of learners referencing this team
        self.outcomes =  dict(outcomes)     # scores at various tasks
        self._id = uuid.uuid4()
        self.preference = None
        self.fitness = fitness
        self.sequence = []

        if isinstance(initParams, dict): self.genCreate = initParams["generation"]
        elif isinstance(initParams, int): self.genCreate = initParams    

    def __hash__(self):
        return int(self._id)

    def _mutation_mutate(self, probability, mutateParams, teams):
        mutated_learners = {}
        '''
         This original learners thing is important, otherwise may mutate learners that we just added through mutation. 
         This breaks reference tracking because it results in 'ghost learners' that were created during mutation, added themselves 
         to inLearners in the teams they pointed to, but them were mutated out before being tracked by the trainer. So you end up
         with teams hold a record in their inLearners to a learner that doesn't exist
        '''
        original_learners = list(self.learners)
        new_learners = []
        for learner in original_learners:
            if flip(probability):

                # If we only have one learner with an atomic memory and the current learner is it
                if self.numAtomicActions() == 1 and learner.isActionAtomic():
                    pActAtom0 = 1.1 # Ensure their memory remains atomic
                else:
                    # Otherwise let there be a probability that the learner's memory is atomic as defined in the mutate params
                    pActAtom0 = mutateParams['pActAtom']

                #print("Team {} creating learner".format(self.id))
                # Create a new new learner 
                newLearner = self.__class__.Learner(
                    program=learner.program, 
                    actionObj=learner.actionObj, 
                    numRegisters=len(learner.registers), 
                    initParams=mutateParams,
                    frameNum=learner.frameNum
                )
                new_learners.append(newLearner)
                # Add the mutated learner to our learners
                # Must add before mutate so that the new learner has this team in its inTeams
                self.addLearner(newLearner)


                # mutate it
                newLearner.mutate(mutateParams, self, teams, pActAtom0)
                # Remove the existing learner from the team
                self.removeLearner(learner)

                mutated_learners[learner._id] = newLearner._id

      
        return mutated_learners, new_learners

    def act(self,
            state,
            visited:list,
            actVars={"frameNum":1},
            path_trace=None
        ):
        # If we've already visited me, throw an exception
        assert not self._id in visited, f"Already visited team {self.id}"
        # Add this team's id to the list of visited ids
        visited.append(self._id)
        if len(self.learners)==0:
            self.addLearner(self.__class__.Learner())


        valid_learners = [lrnr for lrnr in self.learners if lrnr.isActionAtomic() or lrnr.getActionTeam()._id not in visited]
        
        if len(valid_learners)==0: 

            mutate_learner = random.choice(self.learners)
            clone = mutate_learner.clone
            if not clone.isActionAtomic():
                clone.actionObj.teamAction.inLearner.remove(clone._id)
            clone.actionObj.mutate()

            self.addLearner(clone)
            valid_learners.append(clone)


        top_learner = max(valid_learners, key=lambda lrnr: lrnr.bid(state, actVars=actVars))

    
        # If we're tracing this path
        if path_trace != None:
            
            last_segment = path_trace[-1] if len(path_trace) != 0 else None

            # Create our path segment
            path_segment =  {
                'team_id': self.id,
                'top_learner': top_learner.id,
                'top_bid': top_learner.bid(state, actVars=actVars),
                'top_action': top_learner.actionObj.actionCode if top_learner.isActionAtomic() else top_learner.actionObj.teamAction._id,
                'depth': last_segment['depth'] + 1 if last_segment != None else 0,# Record path depth
                'bids': []
            }

            # Populate bid values
            for cursor in valid_learners:
                path_segment['bids'].append({
                    'learner_id': cursor.id,
                    'bid': cursor.bid(state, actVars=actVars),
                    'memory': cursor.actionObj.actionCode if cursor.isActionAtomic() else cursor.actionObj.teamAction._id
                })

            # Append our path segment to the trace
            path_trace.append(path_segment)

        return top_learner.getAction(state, visited=visited, actVars=actVars, path_trace=path_trace)

    def addLearner(self, learner): 
        self.learners.append(learner)
        learner.inTeams.append(self._id) # Add this team's id to the list of teams that reference the learner

        return True

    def removeLearner(self, learner): 
        # only delete if actually in this team
        '''
        TODO log the attempt to remove a learner that doesn't appear in this team
        '''
        if learner not in self.learners:
            raise Exception("Attempted to remove a learner ({}) not referenced by team {}".format(
            learner._id, self._id
        ))

        # Find the learner to remove
        to_remove = [cursor for  cursor in self.learners if cursor == learner]
        if len(to_remove) != 1:
            raise Exception("Duplicate learner detected during team.removeLearner. {} duplicates".format(len(to_remove)))
        to_remove = to_remove[0]

        # Build a new list of learners containing only learners that are not the learner
        self.learners = [cursor for cursor in self.learners if cursor != learner ]

        # Remove our id from the learner's inTeams
        # NOTE: Have to do this after removing the learner otherwise, removal will fail 
        # since the learner's inTeams will not match 
        to_remove.inTeams.remove(self._id)

    def removeLearners(self): 
        for learner in self.learners:
            learner.inTeams.remove(self._id)

        del self.learners[:]

    def mutate(self, mutateParams, allLearners, teams):
        '''
        With rampant mutations every mutateParams["rampantGen"] generations we do X extra
        iterations of mutation. Where X is a random number between mutateParams["rampantMin"] 
        and mutateParams["rampantMax"].
        '''
        # Throw an error if rampantMin is undefined but 

        # Throw an error if rampantMin > rampant Max
        if mutateParams['rampantGen'] != 0 and mutateParams['rampantMin'] > mutateParams['rampantMax']:
            raise Exception("Min rampant iterations is greater than max rampant iterations!", mutateParams)
        
        if (mutateParams["rampantGen"] > 0 and # The rapantGen cannot be 0, as x mod 0 is undefined
            mutateParams["generation"] % mutateParams["rampantGen"] == 0 and # Determine if this is a rampant generation
            mutateParams["generation"] > mutateParams["rampantGen"]  # Rampant generations cannot occur before generation passes rampantGen
            ): 
            rampantReps = random.randrange(mutateParams["rampantMin"], mutateParams["rampantMax"]) if mutateParams['rampantMin'] < mutateParams['rampantMax'] else mutateParams['rampantMin']
        else:
            rampantReps = 1

        # increase diversity by repeating mutations

        mutation_delta = {}
        new_learners = []

        for i in range(rampantReps):
            #print("i/rampant reps:  {}/{} ".format(i, rampantReps))
            # delete some learners
            '''
            TODO log mutation deltas...
            '''
            deleted_learners = self._mutation_delete(mutateParams["pLrnDel"])

            # Create a selection pool from which to add learners to this team
            seed = self.preference.learners if self.preference in teams else allLearners
            # Filter out learners that already belong to this team
            selection_pool = list(filter(lambda x: x not in self.learners, seed))
            
            # Filter out learners that point to this team
            selection_pool = list(filter(lambda x: x._id not in self.inLearners, selection_pool))

            # Filter out learners we just deleted
            selection_pool = list(filter(lambda x: x not in deleted_learners, selection_pool))
            
            added_learners = self._mutation_add(mutateParams["pLrnAdd"], mutateParams["maxTeamSize"], selection_pool)

            # give chance to mutate all learners
            mutated_learners, mutation_added_learners = self._mutation_mutate(mutateParams["pLrnMut"], mutateParams, teams)
            new_learners += mutation_added_learners

            # Compile mutation_delta for this iteration
            mutation_delta[i] = {} 
            mutation_delta[i]['deleted_learners'] = deleted_learners
            mutation_delta[i]['added_learners'] = added_learners
            mutation_delta[i]['mutated_learners'] = mutated_learners

        for cursor in new_learners:
            if cursor in self.learners:
                new_learners.remove(cursor)

        for cursor in new_learners:
            if len(cursor.inTeams) == 0 and not cursor.isActionAtomic():
                cursor.actionObj.teamAction.inLearners.remove(cursor._id)

        # return the number of iterations of mutation
        return rampantReps, mutation_delta, new_learners

    @property
    def id(self):
        return str(self._id)

class Team1_3_1(Team1_3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_3_1
            cls._instance = True
            cls.Learner = Learner1_3_1
        return super().__new__(cls, *args, **kwargs)

class Team1_3_2(Team1_3):

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner1_3_2
            cls._instance = True
            cls.Learner = Learner1_3_2
        return super().__new__(cls, *args, **kwargs)

class Team2(_Team):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner2

            cls._instance = True
            cls.Learner = Learner2

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
            learners: list = [], 
            inLearners: list = [], 
            outcomes: dict = { 'task': 0., 'survive':1., 'reward':0., 'inheritance':0. }, 
            fitness: float = 0, 
            initParams: int or dict = 0
        ):
        super().__init__(learners, inLearners, outcomes, fitness, initParams)

    def _mutation_delete(self, probability):

            original_probability = float(probability)

            if probability == 0.0:
                return []

            if probability >= 1.0: 
                raise Exception("pLrnDel is greater than or equal to 1.0!")

            # Freak out if we don't have an atomic action
            if self.numAtomicMemories() < 1: 
                raise Exception("Less than one atomic memory in team! This shouldn't happen", self)


            deleted_learners = []

            # delete some learners
            while flip(probability) and len(self.learners) > 2: # must have >= 2 learners
                probability *= original_probability # decrease next chance
                # If we have more than one learner with an atomic action pick any learner to delete
                if self.numAtomicMemories() > 1:
                    learner = random.choice(self.learners)
                else: 
                    # Otherwise if we only have one, filter it out and pick from the remaining learners
                    valid_choices = list(filter(lambda x: not x.isMemoryAtomic(), self.learners)) # isActionAtomic以外から削除を決定。
                    learner = random.choice(valid_choices)

                deleted_learners.append(learner)
                self.removeLearner(learner)

            return deleted_learners

    def _mutation_mutate(self, probability, mutateParams, teams):
        mutated_learners = {}
        '''
         This original learners thing is important, otherwise may mutate learners that we just added through mutation. 
         This breaks reference tracking because it results in 'ghost learners' that were created during mutation, added themselves 
         to inLearners in the teams they pointed to, but them were mutated out before being tracked by the trainer. So you end up
         with teams hold a record in their inLearners to a learner that doesn't exist
        '''
        original_learners = list(self.learners)
        new_learners = []
        for learner in original_learners:
            if flip(probability):

                # If we only have one learner with an atomic memory and the current learner is it
                if self.numAtomicMemories() == 1 and learner.isMemoryAtomic():
                    pMemAtom0 = 1.1 # Ensure their memory remains atomic
                else:
                    # Otherwise let there be a probability that the learner's memory is atomic as defined in the mutate params
                    pMemAtom0 = mutateParams['pMemAtom']

                #print("Team {} creating learner".format(self.id))
                # Create a new new learner 
                newLearner = self.__class__.Learner(
                    program=learner.program, 
                    memoryObj=learner.memoryObj, 
                    numRegisters=len(learner.registers), 
                    initParams=mutateParams,
                    frameNum=learner.frameNum
                )
                new_learners.append(newLearner)
                # Add the mutated learner to our learners
                # Must add before mutate so that the new learner has this team in its inTeams
                self.addLearner(newLearner)


                # mutate it
                newLearner.mutate(mutateParams, self, teams, pMemAtom0)
                # Remove the existing learner from the team
                self.removeLearner(learner)

                mutated_learners[str(learner.id)] = str(newLearner.id)

      
        return mutated_learners, new_learners

    def image(self,
        act,
        state,
        visited:list,
        memVars={"frameNum":1},
        path_trace=None
    ):
        # If we've already visited me, throw an exception
        assert not str(self.id) in visited, f"Already visited team {str(self.id)}"
        # Add this team's id to the list of visited ids
        visited.append(str(self.id)) 
        if len(self.learners)==0:
            self.addLearner(self.__class__.Learner())


        valid_learners = [lrnr for lrnr in self.learners if lrnr.isMemoryAtomic() or str(lrnr.getMemoryTeam().id) not in visited]
        
        if len(valid_learners)==0: 

            mutate_learner = random.choice(self.learners)
            clone = mutate_learner.clone
            if not clone.isMemoryAtomic():
                clone.memoryObj.teamMemory.inLearner.remove(str(clone.id))
            clone.memoryObj.mutate()

            self.addLearner(clone)
            valid_learners.append(clone)


        top_learner = max(valid_learners, key=lambda lrnr: lrnr.bid(act, state, memVars=memVars))

    
        # If we're tracing this path
        if path_trace != None:
            
            last_segment = path_trace[-1] if len(path_trace) != 0 else None

            # Create our path segment
            path_segment =  {
                'team_id': str(self.id),
                'top_learner': str(top_learner.id),
                'top_bid': top_learner.bid(act, state, memVars=memVars),
                'top_memory': top_learner.memoryObj.memoryCode if top_learner.isMemoryAtomic() else str(top_learner.memoryObj.teamMemory.id),
                'depth': last_segment['depth'] + 1 if last_segment != None else 0,# Record path depth
                'bids': []
            }

            # Populate bid values
            for cursor in valid_learners:
                path_segment['bids'].append({
                    'learner_id': str(cursor.id),
                    'bid': cursor.bid(act, state, memVars=memVars),
                    'memory': cursor.memoryObj.memoryCode if cursor.isMemoryAtomic() else str(cursor.memoryObj.teamMemory.id)
                })

            # Append our path segment to the trace
            path_trace.append(path_segment)

        return top_learner.getImage(act, state, visited=visited, memVars=memVars, path_trace=path_trace)

    def numAtomicMemories(self):
        num = 0
        for lrnr in self.learners:
            if lrnr.isMemoryAtomic():
                num += 1

        return num

    def mutate(self, mutateParams, allLearners, teams):
        # Throw an error if rampantMin > rampant Max
        if mutateParams['rampantGen'] != 0 and mutateParams['rampantMin'] > mutateParams['rampantMax']:
            raise Exception("Min rampant iterations is greater than max rampant iterations!", mutateParams)
        
        if (mutateParams["rampantGen"] > 0 and # The rapantGen cannot be 0, as x mod 0 is undefined
            mutateParams["generation"] % mutateParams["rampantGen"] == 0 and # Determine if this is a rampant generation
            mutateParams["generation"] > mutateParams["rampantGen"]  # Rampant generations cannot occur before generation passes rampantGen
            ): 
            rampantReps = random.randrange(mutateParams["rampantMin"], mutateParams["rampantMax"]) if mutateParams['rampantMin'] < mutateParams['rampantMax'] else mutateParams['rampantMin']
        else:
            rampantReps = 1

        # increase diversity by repeating mutations

        mutation_delta = {}
        new_learners = []

        for i in range(rampantReps):
            #print("i/rampant reps:  {}/{} ".format(i, rampantReps))
            # delete some learners
            '''
            TODO log mutation deltas...
            '''
            deleted_learners = self._mutation_delete(mutateParams["pLrnDel"])

            # Create a selection pool from which to add learners to this team
            
            # Filter out learners that already belong to this team
            selection_pool = list(filter(lambda x: x not in self.learners, allLearners))
            
            # Filter out learners that point to this team
            selection_pool = list(filter(lambda x: str(x.id) not in self.inLearners, selection_pool))

            # Filter out learners we just deleted
            selection_pool = list(filter(lambda x: x not in deleted_learners, selection_pool))
            
            added_learners = self._mutation_add(mutateParams["pLrnAdd"], mutateParams["maxTeamSize"], selection_pool)

            # give chance to mutate all learners
            mutated_learners, mutation_added_learners = self._mutation_mutate(mutateParams["pLrnMut"], mutateParams, teams)
            new_learners += mutation_added_learners

            # Compile mutation_delta for this iteration
            mutation_delta[i] = {} 
            mutation_delta[i]['deleted_learners'] = deleted_learners
            mutation_delta[i]['added_learners'] = added_learners
            mutation_delta[i]['mutated_learners'] = mutated_learners

        for cursor in new_learners:
            if cursor in self.learners:
                new_learners.remove(cursor)

        for cursor in new_learners:
            if len(cursor.inTeams) == 0 and not cursor.isMemoryAtomic():
                cursor.memoryObj.teamMemory.inLearners.remove(str(cursor.id))

        # return the number of iterations of mutation
        return rampantReps, mutation_delta, new_learners

class Team2_1(Team2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner2_1
            cls._instance = True
            cls.Learner = Learner2_1

        return super().__new__(cls, *args, **kwargs)

class Team2_2(Team2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner2_2
            cls._instance = True
            cls.Learner = Learner2_2
            
        return super().__new__(cls, *args, **kwargs)

class Team2_3(Team2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner2_3
            cls._instance = True
            cls.Learner = Learner2_3
            
        return super().__new__(cls, *args, **kwargs)

    def __init__(self,
            learners:list = [],
            inLearners:list = [],
            outcomes:dict = {'task':0.},
            fitness:float = 0.0,
            initParams:int or dict=0
        ): 
        self.learners = list(learners)
        self.inLearners = list(inLearners)  # ids of learners referencing this team
        self.outcomes =  dict(outcomes)     # scores at various tasks
        self._id = uuid.uuid4()
        self.preference = None
        self.fitness = fitness
        self.sequence = []
        self.extinction = 1.
        if isinstance(initParams, dict): self.genCreate = initParams["generation"]
        elif isinstance(initParams, int): self.genCreate = initParams    

    def __hash__(self):
        return int(self._id)

    def _mutation_mutate(self, probability, mutateParams, teams):
        mutated_learners = {}
        '''
         This original learners thing is important, otherwise may mutate learners that we just added through mutation. 
         This breaks reference tracking because it results in 'ghost learners' that were created during mutation, added themselves 
         to inLearners in the teams they pointed to, but them were mutated out before being tracked by the trainer. So you end up
         with teams hold a record in their inLearners to a learner that doesn't exist
        '''
        original_learners = list(self.learners)
        new_learners = []
        for learner in original_learners:
            if flip(probability):

                # If we only have one learner with an atomic memory and the current learner is it
                if self.numAtomicMemories() == 1 and learner.isMemoryAtomic():
                    pMemAtom0 = 1.1 # Ensure their memory remains atomic
                else:
                    # Otherwise let there be a probability that the learner's memory is atomic as defined in the mutate params
                    pMemAtom0 = mutateParams['pMemAtom']

                #print("Team {} creating learner".format(self.id))
                # Create a new new learner 
                newLearner = self.__class__.Learner(
                    program=learner.program, 
                    memoryObj=learner.memoryObj, 
                    numRegisters=len(learner.registers), 
                    initParams=mutateParams,
                    frameNum=learner.frameNum
                )
                new_learners.append(newLearner)
                # Add the mutated learner to our learners
                # Must add before mutate so that the new learner has this team in its inTeams
                self.addLearner(newLearner)


                # mutate it
                newLearner.mutate(mutateParams, self, teams, pMemAtom0)
                # Remove the existing learner from the team
                self.removeLearner(learner)

                mutated_learners[learner._id] = newLearner._id

      
        return mutated_learners, new_learners

    def image(self,
        state,
        visited:list,
        memVars={"frameNum":1},
        path_trace=None
    ):
        # If we've already visited me, throw an exception
        assert not self.id in visited, f"Already visited team {self.id}"
        # Add this team's id to the list of visited ids
        visited.append(self.id) 
        if len(self.learners)==0:
            self.addLearner(self.__class__.Learner())

        self.extinction*=0.9

        valid_learners = [lrnr for lrnr in self.learners if lrnr.isMemoryAtomic() or lrnr.getMemoryTeam().id not in visited]
        
        if len(valid_learners)==0: 

            mutate_learner = random.choice(self.learners)
            clone = mutate_learner.clone
            if not clone.isMemoryAtomic():
                clone.memoryObj.teamMemory.inLearner.remove(clone.id)
            clone.memoryObj.mutate()

            self.addLearner(clone)
            valid_learners.append(clone)


        top_learner = max(valid_learners, key=lambda lrnr: lrnr.bid(state, memVars=memVars))

    
        # If we're tracing this path
        if path_trace != None:
            
            last_segment = path_trace[-1] if len(path_trace) != 0 else None

            # Create our path segment
            path_segment =  {
                'team_id': self.id,
                'top_learner': top_learner.id,
                'top_bid': top_learner.bid(state, memVars=memVars),
                'top_memory': top_learner.memoryObj.memoryCode if top_learner.isMemoryAtomic() else top_learner.memoryObj.teamMemory.id,
                'depth': last_segment['depth'] + 1 if last_segment != None else 0,# Record path depth
                'bids': []
            }

            # Populate bid values
            for cursor in valid_learners:
                path_segment['bids'].append({
                    'learner_id': cursor.id,
                    'bid': cursor.bid(state, memVars=memVars),
                    'memory': cursor.memoryObj.memoryCode if cursor.isMemoryAtomic() else cursor.memoryObj.teamMemory.id
                })

            # Append our path segment to the trace
            path_trace.append(path_segment)

        return top_learner.getImage(state, visited=visited, memVars=memVars, path_trace=path_trace)

    def addSequence(self):
        imgs = np.random.randint(self.sequence)
        for img in imgs:
            image = abstract(self.sequence[img])
            self.addLearner(self.__class__.Learner(memoryObj=image))

    def addLearner(self, learner): 
        self.learners.append(learner)
        learner.inTeams.append(self.id) # Add this team's id to the list of teams that reference the learner

        return True

    def removeLearner(self, learner): 
        # only delete if actually in this team
        '''
        TODO log the attempt to remove a learner that doesn't appear in this team
        '''
        if learner not in self.learners:
            raise Exception("Attempted to remove a learner ({}) not referenced by team {}".format(
            learner.id, self.id
        ))

        # Find the learner to remove
        to_remove = [cursor for  cursor in self.learners if cursor == learner]
        if len(to_remove) != 1:
            raise Exception("Duplicate learner detected during team.removeLearner. {} duplicates".format(len(to_remove)))
        to_remove = to_remove[0]

        # Build a new list of learners containing only learners that are not the learner
        self.learners = [cursor for cursor in self.learners if cursor != learner ]

        # Remove our id from the learner's inTeams
        # NOTE: Have to do this after removing the learner otherwise, removal will fail 
        # since the learner's inTeams will not match 
        to_remove.inTeams.remove(self.id)

    def removeLearners(self): 
        for learner in self.learners:
            learner.inTeams.remove(self.id)

        del self.learners[:]

    def mutate(self, mutateParams, allLearners, teams):
        '''
        With rampant mutations every mutateParams["rampantGen"] generations we do X extra
        iterations of mutation. Where X is a random number between mutateParams["rampantMin"] 
        and mutateParams["rampantMax"].
        '''
        # Throw an error if rampantMin is undefined but 

        # Throw an error if rampantMin > rampant Max
        if mutateParams['rampantGen'] != 0 and mutateParams['rampantMin'] > mutateParams['rampantMax']:
            raise Exception("Min rampant iterations is greater than max rampant iterations!", mutateParams)
        
        if (mutateParams["rampantGen"] > 0 and # The rapantGen cannot be 0, as x mod 0 is undefined
            mutateParams["generation"] % mutateParams["rampantGen"] == 0 and # Determine if this is a rampant generation
            mutateParams["generation"] > mutateParams["rampantGen"]  # Rampant generations cannot occur before generation passes rampantGen
            ): 
            rampantReps = random.randrange(mutateParams["rampantMin"], mutateParams["rampantMax"]) if mutateParams['rampantMin'] < mutateParams['rampantMax'] else mutateParams['rampantMin']
        else:
            rampantReps = 1

        # increase diversity by repeating mutations

        mutation_delta = {}
        new_learners = []

        for i in range(rampantReps):
            #print("i/rampant reps:  {}/{} ".format(i, rampantReps))
            # delete some learners
            '''
            TODO log mutation deltas...
            '''
            deleted_learners = self._mutation_delete(mutateParams["pLrnDel"])

            # Create a selection pool from which to add learners to this team
            seed = self.preference.learners if self.preference in teams else allLearners
            # Filter out learners that already belong to this team
            selection_pool = list(filter(lambda x: x not in self.learners, seed))
            
            # Filter out learners that point to this team
            selection_pool = list(filter(lambda x: x.id not in self.inLearners, selection_pool))

            # Filter out learners we just deleted
            selection_pool = list(filter(lambda x: x not in deleted_learners, selection_pool))
            
            added_learners = self._mutation_add(mutateParams["pLrnAdd"], mutateParams["maxTeamSize"], selection_pool)

            # give chance to mutate all learners
            mutated_learners, mutation_added_learners = self._mutation_mutate(mutateParams["pLrnMut"], mutateParams, teams)
            new_learners += mutation_added_learners

            # Compile mutation_delta for this iteration
            mutation_delta[i] = {} 
            mutation_delta[i]['deleted_learners'] = deleted_learners
            mutation_delta[i]['added_learners'] = added_learners
            mutation_delta[i]['mutated_learners'] = mutated_learners

        for cursor in new_learners:
            if cursor in self.learners:
                new_learners.remove(cursor)

        for cursor in new_learners:
            if len(cursor.inTeams) == 0 and not cursor.isMemoryAtomic():
                cursor.memoryObj.teamMemory.inLearners.remove(cursor.id)

        # return the number of iterations of mutation
        return rampantReps, mutation_delta, new_learners

    @property
    def id(self):
        return str(self._id)

class Team3(Team2_3):
    """
       TODO: integration of ActionObj and MemoryObj
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.learner import Learner3
            cls._instance = True
            cls.Learner = Learner3
            
        return super().__new__(cls, *args, **kwargs)
