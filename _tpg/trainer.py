from _tpg.agent import Agent, Agent1, Agent3, Agent2
from _tpg.team import Team, Team1, Team3, Team2
from _tpg.learner import Learner, Learner1, Learner3, Learner2
from _tpg.program import Program, Program1, Program3, Program2
from _tpg.action_object import ActionObject, ActionObject1, ActionObject3, ActionObject2
from _tpg.memory_object import MemoryObject
from _tpg.configuration import configurer
# from _tpg.emulator import Emulator
import random
import pickle
import numpy as np
# import time
# import multiprocessing as mp

def breakpoint(*_print):
    import sys
    print(_print)
    sys.exit()
"""
Functionality for actually growing TPG and evolving it to be functional.
"""
class Trainer:
    def __init__(self, 
        actions=2, 
        teamPopSize:int=1000,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pActMut:float=0.1,                  # *
        pActAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memType=None, 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4
        ):


        '''
        Validate inputs
        '''
        int_greater_than_zero = {
            "teamPopSize": teamPopSize,
            "inputSize": inputSize,
            "nRegisters": nRegisters,
            "initMaxTeamSize": initMaxTeamSize,
            "initMaxProgSize": initMaxProgSize,
        }

        for entry in int_greater_than_zero.items(): self._must_be_integer_greater_than_zero(entry[0], entry[1])

        # Validate doElites
        if type(doElites) is not bool:  raise Exception("Invalid doElites")

        # Validate rootBasedPop
        if type(rootBasedPop) is not bool:  raise Exception("Invalid rootBasedPop")

        # Validate Traversal
        if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

        # Validate Probability parameters
        probabilities = {
            "pLrnDel": pLrnDel,
            "pLrnAdd": pLrnAdd,
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut,
            "pActMut": pActMut,
            "pActAtom": pActAtom,
            "pInstDel": pInstDel,
            "pInstAdd": pInstAdd,
            "pInstSwp": pInstSwp,
            "pInstMut": pInstMut
        }


        for entry in probabilities.items(): self._validate_probability(entry[0],entry[1])

        # Validate rampancy
        if (
            len(rampancy) != 3 or
            rampancy[0] < 0 or
            rampancy[2] < rampancy[1] or 
            rampancy[1] < 0 or
            rampancy[2] < 0
        ): raise Exception("Invalid rampancy parameter!", rampancy)

        # store all necessary params

        # first store actions properly
        self.doReal = self._setUpActions(actions)

        # population params
        self.teamPopSize = teamPopSize
        # whether population size is based on root teams or all teams
        self.rootBasedPop = rootBasedPop
        self.gap = gap # portion of root teams to remove each generation

        # input to agent (must be at-least size of input/state from environment)
        self.inputSize = inputSize # defaulted to number of Atari screen pixels
        # number of local memory registers each learner will have
        self.nRegisters = nRegisters

        # params for initializing evolution
        self.initMaxTeamSize = initMaxTeamSize # size of team = # of learners
        self.initMaxProgSize = initMaxProgSize # size of program = # of instructions

        # max team size possible throughout evolution
        self.maxTeamSize = maxTeamSize

        # params for continued evolution
        self.pLrnDel = pLrnDel
        self.pLrnAdd = pLrnAdd
        self.pLrnMut = pLrnMut
        self.pProgMut = pProgMut
        self.pActMut = pActMut
        self.pActAtom = pActAtom
        self.pInstDel = pInstDel
        self.pInstAdd = pInstAdd
        self.pInstSwp = pInstSwp
        self.pInstMut = pInstMut

        # whether to keep elites
        self.doElites = doElites

        if memType == "None":   self.memType = None
        self.memType = memType
        self.memMatrixShape = memMatrixShape

        self.rampancy = rampancy

        self.operationSet = operationSet

        self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nActRegisters = nActRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {}
        self.actVars:       dict = {}
        self.functionsDict: dict = {}
        self.nOperations = None



        # configure tpg functions and variable appropriately now
        configurer.configure(self, Trainer, Agent, Team, Learner, ActionObject, Program, memType is not None, memType, self.doReal, operationSet, traversal)

        #print(self.mutateParams)
        #print(self.functionsDict)
        #print(1/0)
        self._initializePopulations()

    '''
    Validation Method
    '''
    def _must_be_integer_greater_than_zero(self, name, value):
        if type(value) is not int or value <= 0:
            raise Exception(name + " must be integer greater than zero. Got " + str(value), name, value)

    '''
    Validation Method
    '''
    def _validate_probability(self, name,  value):
        if type(value) is not float or value > 1.0 or value < 0.0:
            raise Exception(name + " is a probability, it must not be greater than 1.0 or less than 0.0", name, value)

    """
    Sets up the actions properly, splitting action codes, and if needed, action
    lengths. Returns whether doing real actions.
    """
    def _setUpActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes = range(actions)
            doReal = False
        else: # list of lengths of each action
            # some may be real actions
            self.actionLengths = list(actions)
            self.actionCodes = range(len(actions))
            doReal = True

        return doReal
    
    def resetActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes:list = range(actions)
            self.doReal = False
        else: # list of lengths of each action
            # some may be real actions
            self.actionCodes:list = range(len(actions))
            self.actionLengths = list(actions)
            self.doReal = True
        
        if self.doReal: self.nActRegisters = max(max(self.actionLengths), self.nActRegisters)
        self.nActRegisters = self.nActRegisters
        configurer.configure(self, Trainer, Agent, Team, Learner, ActionObject, Program, 
            self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)

    """
    Initializes a popoulation of teams and learners generated randomly with only
    atomic actions.
    """
    def _initializePopulations(self):
        for i in range(self.teamPopSize):
            # create 2 unique actions and learners
            a1,a2 = random.sample(range(len(self.actionCodes)), 2)

            l1 = Learner(self.mutateParams,
                        program=Program(maxProgramLength=self.initMaxProgSize,
                                         nOperations=self.nOperations,
                                         nDestinations=self.nRegisters,
                                         inputSize=self.inputSize,
                                         initParams=self.mutateParams),
                        actionObj=ActionObject(action=a1, initParams=self.mutateParams),
                        numRegisters=self.nRegisters)
            l2 = Learner(self.mutateParams,
                        program=Program(maxProgramLength=self.initMaxProgSize,
                                         nOperations=self.nOperations,
                                         nDestinations=self.nRegisters,
                                         inputSize=self.inputSize,
                                         initParams=self.mutateParams),
                        actionObj=ActionObject(action=a2, initParams=self.mutateParams),
                        numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for i in range(moreLearners):
                # select action
                act = random.choice(range(len(self.actionCodes)))

                # create new learner
                learner = Learner(initParams=self.mutateParams,
                            program=Program(maxProgramLength=self.initMaxProgSize,
                                             nOperations=self.nOperations,
                                             nDestinations=self.nRegisters,
                                             inputSize=self.inputSize,
                                             initParams=self.mutateParams),
                            actionObj=ActionObject(action=act, initParams=self.mutateParams),
                            numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    """
    Gets rootTeams/agents. Sorts decending by sortTasks, and skips individuals
    who don't have scores for all skipTasks.
    """
    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[]):
        # remove those that get skipped
        rTeams = [team for team in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in team.outcomes for task in skipTasks)]

        if len(sortTasks) == 0: # just get all
            return [Agent(team, self.functionsDict, num=i, actVars=self.actVars)
                    for i,team in enumerate(rTeams)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [Agent(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]

            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [Agent(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    """ 
    Gets the single best team at the given task, regardless of if its root or not.
    """
    def getEliteAgent(self, task):
        
        teams = [t for t in self.teams if task in t.outcomes]

        return Agent(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                     self.functionsDict, num=0, actVars=self.actVars)

    """
    Apply saved scores from list to the agents.
    """
    def applyScores(self, scores): # used when multiprocessing
        for score in scores:
            for rt in self.rootTeams:
                if score[0] == rt.id:
                    for task, outcome in score[1].items():
                        rt.outcomes[task] = outcome
                    break # on to next score

        return self.rootTeams

    """
    Evolve the populations for improvements.
    """
    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self._scoreIndividuals(tasks, multiTaskType=multiTaskType,
                doElites=self.doElites) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch() # set up for next generation
        #self.validate_graph() # validate the tpg (for debug only)
    """
    Assigns a fitness to each agent based on performance at the tasks. Assigns
    fitness values, or just returns sorted root teams.
    """
    def _scoreIndividuals(self, tasks, multiTaskType='min', doElites=True):
        # handle generation of new elites, typically just done in evolution
        if doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([team for team in self.rootTeams],
                                        key=lambda t: t.outcomes[task]))

        if len(tasks) == 1: # single fitness
            for team in self.rootTeams:
                team.fitness = team.outcomes[tasks[0]]
        else: # multi fitness
            # assign fitness to each agent based on tasks and score type
            if 'pareto' not in multiTaskType or 'lexicase' not in multiTaskType:
                self.simpleScorer(tasks, multiTaskType=multiTaskType)
            elif multiTaskType == 'paretoDominate':
                self.paretoDominateScorer(tasks)
            elif multiTaskType == 'paretoNonDominated':
                self.paretoNonDominatedScorer(tasks)
            elif multiTaskType == 'lexicaseStatic':
                self.lexicaseStaticScorer(tasks)
            elif multiTaskType == 'lexicaseDynamic':
                self.lexicaseDynamicScorer(tasks)

    """
    Gets either the min, max, or average score from each individual for ranking.
    """
    def simpleScorer(self, tasks, multiTaskType='min'):
        # first find min and max in each task
        mins = []
        maxs = []
        for task in tasks:
            mins.append(min([team.outcomes[task] for team in self.rootTeams]))
            maxs.append(max([team.outcomes[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                            for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    """
    Rank agents based on how many other agents it dominates
    """
    def paretoDominateScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] >= t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness += 1

    """
    Rank agents based on how many other agents don't dominate it
    """
    def paretoNonDominatedScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] < t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness -= 1

    def lexicaseStaticScorer(self, tasks):
        stasks = list(tasks)
        random.shuffle(stasks)

        for rt in self.rootTeams:
            rt.fitness = rt.outcomes[tasks[0]]


    def lexicaseDynamicScorer(self, tasks):
        pass

    """
    Save some stats on the fitness.
    """
    def _saveFitnessStats(self):
        fitnesses = []
        for rt in self.rootTeams:
            fitnesses.append(rt.fitness)

        self.fitnessStats = {}
        self.fitnessStats['fitnesses'] = fitnesses
        self.fitnessStats['min'] = min(fitnesses)
        self.fitnessStats['max'] = max(fitnesses)
        self.fitnessStats['average'] = sum(fitnesses)/len(fitnesses)

    """
    Gets stats on some task.
    """
    def getTaskStats(self, task):
        scores = []
        for rt in self.rootTeams:
            scores.append(rt.outcomes[task])

        scoreStats = {}
        scoreStats['scores'] = scores
        scoreStats['min'] = min(scores)
        scoreStats['max'] = max(scores)
        scoreStats['average'] = sum(scores)/len(scores)

        return scoreStats

    """
    Select a portion of the root team population to keep according to gap size.
    """
    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness, reverse=True)
        numKeep = len(self.rootTeams) - int(len(self.rootTeams)*self.gap)
        deleteTeams = rankedTeams[numKeep:]
        for team in [t for t in deleteTeams if t not in self.elites]:    
            # remove learners from team and delete team from populations
            if extraTeams is None or team not in extraTeams:
                team.removeLearners()
            self.teams.remove(team)
            self.rootTeams.remove(team)

        orphans = [learner for learner in self.learners if learner.numTeamsReferencing() == 0]

        for cursor in orphans:
            if not cursor.isActionAtomic(): # If the orphan does NOT point to an atomic action
                # Get the team the orphan is pointing to and remove the orphan's id from the team's in learner list
                cursor.actionObj.teamAction.inLearners.remove(str(cursor.id))

        # Finaly, purge the orphans
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0]
                

    """
    Generates new rootTeams based on existing teams.
    """
    def _generate(self, extraTeams=None):

        # extras who are already part of the team population
        protectedExtras = []
        extrasAdded = 0

        # add extras into the population
        if extraTeams is not None:
            for team in extraTeams:
                if team not in self.teams:
                    self.teams.append(team)
                    extrasAdded += 1
                else:
                    protectedExtras.append(team)

        oLearners = list(self.learners)
        oTeams = list(self.teams)

        # update generation in mutateParams
        self.mutateParams["generation"] = self.generation

        # get all the current root teams to be parents
        while (len(self.teams) < self.teamPopSize + extrasAdded or
                (self.rootBasedPop and self.countRootTeams() < self.teamPopSize)):
            # get parent root team, and child to be based on that
            parent = random.choice(self.rootTeams)
            child = Team(initParams=self.mutateParams)

            # child starts just like parent
            for learner in parent.learners:
                child.addLearner(learner)

            # then mutates
            child.mutate(self.mutateParams, oLearners, oTeams)

            self.teams.append(child)

        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    """
    Finalize populations and prepare for next generation/epoch.
    """
    def _nextEpoch(self):
        # add in newly added learners, and decide root teams
        self.rootTeams = []
        for team in self.teams:
            # add any new learners to the population
            for learner in team.learners:
                if learner not in self.learners:
                    #print("Adding {} to trainer learners".format(learner.id))
                    self.learners.append(learner)

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        self.generation += 1

    """
    Removes hitchhikers, learners that are never used, except for the last atomic action on the team.
    teamLearnerVisists is a dict with team keys and values represending the learners that are
    actually visited on the team. Any learner on a team not in this list gets deleted.
    Evolve should be called right after to properly remove the learners from the population.
    """
    def removeHitchhikers(self, teams, visitedLearners):
        learnersRemoved = []
        teamsAffected = []

        for i, team in enumerate(teams):
            affected = False
            for learner in team.learners:
                # only remove if non atomic, or atomic and team has > 1 atomic actions
                if learner not in visitedLearners[i] and (
                        not learner.isActionAtomic() or 
                            (learner.isActionAtomic() and team.numAtomicActions() > 1)):
                    affected = True
                    learnersRemoved.append(learner)
                    team.removeLearner(learner)

            if affected:    teamsAffected.append(team)

        return learnersRemoved, teamsAffected
    
    '''
    Go through all teams and learners and make sure their inTeams/inLearners correspond with 
    their team.learner/teamActions as expected.
    '''
    def validate_graph(self):
        print("Validating graph")

        print("Checking for broken learners")
        for cursor in self.learners:
            if cursor.isActionAtomic() and cursor.actionObj.actionCode == None:
                print("{} is an action atomic learner with no actionCode!".format(str(cursor.id)))
            if cursor.actionObj.teamAction == None and cursor.actionObj.actionCode == None:
                print("{} has no action!".format(str(cursor.id)))

        learner_map = {}
        for cursor in self.learners:
            if str(cursor.id) not in learner_map:
                learner_map[str(cursor.id)] = cursor.inTeams
            else:
                raise Exception("Duplicate learner id in trainer!")
        
        '''
        For every entry in the learner map check that the corresponding team has the learner 
        in its learners 
        '''
        for i,cursor in enumerate(learner_map.items()):
            for expected_team in cursor[1]:
                found_team = False
                for team in self.teams:
                    if str(team.id) == expected_team:
                        found_learner = 0
                        for learner in team.learners:
                            if str(learner.id) == cursor[0]:
                                found_learner += 1
                        if found_learner != 1:
                            print("found_learner = {} for learner {} in team {}".format(found_learner, cursor[0], str(team.id)))
                        found_team = True
                        break
                if found_team == False:
                    print("Could not find expected team {} in trainer".format(expected_team))
            print("learner {} inTeams valid [{}/{}]".format(cursor[0], i, len(learner_map.items())-1))

        '''
        Verify that for every inLearner in a team, the learner exists in the trainer, pointing to that team
        '''
        team_map = {}
        for cursor in self.teams:
            if str(cursor.id) not in team_map:
                team_map[str(cursor.id)] = cursor.inLearners
            else:
                raise Exception("Duplicate team id in trainer!")

        for i,cursor in enumerate(team_map.items()):
            for expected_learner in cursor[1]:
                found_learner = False
                points_to_team = False
                for learner in self.learners:
                    if str(learner.id) == expected_learner:
                        found_learner = True
                        if str(learner.actionObj.teamAction.id) == cursor[0]:
                            points_to_team = True
                            break
                if found_learner == False:
                    print("Could not find learner {} from team {} inLearners in trainer.".format(expected_learner, cursor[0]))
                if points_to_team == False:
                    print("Learner {} does not point to team {}".format(expected_learner, cursor[0]))
            print("team {} inLearners valid [{}/{}]".format(cursor[0], i, len(team_map.items())-1))
    
    """
    Get the number of root teams currently residing in the teams population.
    """
    def countRootTeams(self):
        numRTeams = 0
        for team in self.teams:
            if team.numLearnersReferencing() == 0: numRTeams += 1

        return numRTeams

    def get_graph(self):


        result = {
            "nodes":[],
            "links":[]
        }

        # First add action codes as nodes
        for actionCode in self.actionCodes:
            result["nodes"].append(
                {
                    "id": str(actionCode),   
                    "type": "action"
                }
            )
        
        # Then add teams as nodes
        for team in self.teams:
            result["nodes"].append(
                {
                    "id": str(team.id),
                    "type": "rootTeam" if team in self.rootTeams else "team"
                }
            )

        # Then add learners as nodes
        for learner in self.learners:
            result["nodes"].append(
                {
                    "id": str(learner.id),
                    "type": "learner"
                }
            )


        # Then add links from learners to teams
        for team in self.teams:
            for learner in team.inLearners:
                result["links"].append(
                    {
                        "source": learner,
                        "target": str(team.id)
                    }
                )
        
        # Then add links from teams to learners
        for learner in self.learners:
            for team in learner.inTeams:
                result["links"].append(
                    {
                        "source": team,
                        "target": str(learner.id)
                    }
                )
            
            # Also add links to action codes
            if learner.isActionAtomic():
                result["links"].append(
                    {
                        "source": str(learner.id),
                        "target": str(learner.actionObj.actionCode)
                    }
                )

        # with open("tpg_{}.json".format(self.generation), 'w') as out_file:
        #     json.dump(result, out_file)
        return result

    """
    Function to cleanup anything that may interfere with another trainer run in
    the same thread of execution. Currently just sets tpg module functions to defaults.
    """
    def cleanup(self):
        configurer._configureDefaults(self, Trainer, Agent, Team, Learner, ActionObject, Program)

    """
    Ensures proper functions are in place for all classes. Ran after loading from a file,
    and may need to be ran in other cases such as multiprocessing (though in out typical use
    that is done at the agent level).
    """
    def configFunctions(self):
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
    Save the trainer to the file, saving any class values to the instance.
    """
    def saveToFile(self, fileName):
        pickle.dump(self, open(f'log/{fileName}.pickle', 'wb'))

def loadTrainer(fileName:str):
    trainer = pickle.load(open(f'log/{fileName}.pickle', 'rb'))
    trainer.configFunctions()
    return trainer

class Trainer1:

    def __init__(self, 
        actions=2, 
        teamPopSize:int=1000,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pActMut:float=0.1,                  # *
        pActAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4
    ):


        '''
        Validate inputs
        '''
        int_greater_than_zero = {
            "teamPopSize": teamPopSize,
            "inputSize": inputSize,
            "nRegisters": nRegisters,
            "initMaxTeamSize": initMaxTeamSize,
            "initMaxProgSize": initMaxProgSize,
        }

        for entry in int_greater_than_zero.items(): self._must_be_integer_greater_than_zero(entry[0], entry[1])

        # Validate doElites
        if type(doElites) is not bool:  raise Exception("Invalid doElites")

        # Validate rootBasedPop
        if type(rootBasedPop) is not bool:  raise Exception("Invalid rootBasedPop")

        # Validate Traversal
        if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

        # Validate Probability parameters
        probabilities = {
            "pLrnDel": pLrnDel,
            "pLrnAdd": pLrnAdd,
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut,
            "pActMut": pActMut,
            "pActAtom": pActAtom,
            "pInstDel": pInstDel,
            "pInstAdd": pInstAdd,
            "pInstSwp": pInstSwp,
            "pInstMut": pInstMut
        }


        for entry in probabilities.items(): self._validate_probability(entry[0],entry[1])

        # Validate rampancy
        if (
            len(rampancy) != 3 or
            rampancy[0] < 0 or
            rampancy[2] < rampancy[1] or 
            rampancy[1] < 0 or
            rampancy[2] < 0
        ): raise Exception("Invalid rampancy parameter!", rampancy)

        # store all necessary params

        # first store actions properly
        self.doReal = self._setUpActions(actions)

        # population params
        self.teamPopSize = teamPopSize
        # whether population size is based on root teams or all teams
        self.rootBasedPop = rootBasedPop
        self.gap = gap # portion of root teams to remove each generation

        # input to agent (must be at-least size of input/state from environment)
        self.inputSize = inputSize # defaulted to number of Atari screen pixels
        # number of local memory registers each learner will have
        self.nRegisters = nRegisters

        # params for initializing evolution
        self.initMaxTeamSize = initMaxTeamSize # size of team = # of learners
        self.initMaxProgSize = initMaxProgSize # size of program = # of instructions

        # max team size possible throughout evolution
        self.maxTeamSize = maxTeamSize

        # params for continued evolution
        self.pLrnDel = pLrnDel
        self.pLrnAdd = pLrnAdd
        self.pLrnMut = pLrnMut
        self.pProgMut = pProgMut
        self.pActMut = pActMut
        self.pActAtom = pActAtom
        self.pInstDel = pInstDel
        self.pInstAdd = pInstAdd
        self.pInstSwp = pInstSwp
        self.pInstMut = pInstMut

        # whether to keep elites
        self.doElites = doElites

        if memType == "None":   self.memType = None
        self.memType = memType
        self.memMatrixShape = memMatrixShape

        self.rampancy = rampancy

        self.operationSet = operationSet

        self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nActRegisters = nActRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {}
        self.actVars:       dict = {}
        self.functionsDict: dict = {}
        self.nOperations = None



        # configure tpg functions and variable appropriately now
        configurer.configure1(self, Trainer1, Agent1, Team1, Learner1, ActionObject1, Program1, self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)

        self._initializePopulations()

    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[]):
        # remove those that get skipped
        rTeams = [team for team in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in team.outcomes for task in skipTasks)]

        if len(sortTasks) == 0: # just get all
            return [Agent1(team, self.functionsDict, num=i, actVars=self.actVars)
                    for i,team in enumerate(rTeams)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [Agent1(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]

            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [Agent1(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    def getEliteAgent(self, task):
        
        teams = [t for t in self.teams if task in t.outcomes]

        return Agent1(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                     self.functionsDict, num=0, actVars=self.actVars)

    def applyScores(self, scores): # used when multiprocessing
        for score in scores:
            for rt in self.rootTeams:
                if score[0] == rt.id:
                    for task, outcome in score[1].items():
                        rt.outcomes[task] = outcome
                    break # on to next score

        return self.rootTeams

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
            doElites=self.doElites
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def _must_be_integer_greater_than_zero(self, name, value):
        if type(value) is not int or value <= 0:
            raise Exception(name + " must be integer greater than zero. Got " + str(value), name, value)

    def _validate_probability(self, name,  value):
        if type(value) is not float or value > 1.0 or value < 0.0:
            raise Exception(name + " is a probability, it must not be greater than 1.0 or less than 0.0", name, value)

    def _setUpActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes:list = range(actions)
            doReal = False
        else: # list of lengths of each action
            # some may be real actions
            self.actionCodes:list = range(len(actions))
            self.actionLengths = list(actions)
            doReal = True
            
        return doReal
    
    def resetActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes:list = range(actions)
            self.doReal = False
        else: # list of lengths of each action
            # some may be real actions
            self.actionCodes:list = range(len(actions))
            self.actionLengths = list(actions)
            self.doReal = True
        
        if self.doReal: self.nActRegisters = max(max(self.actionLengths), self.nActRegisters)
        self.nActRegisters = self.nActRegisters
        configurer.configure1(self, Trainer1, Agent1, Team1, Learner1, ActionObject1, Program1, 
            self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)

    def _initializePopulations(self):
        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            a1,a2 = random.sample(range(len(self.actionCodes)), 2)

            l1 = Learner1(
                initParams=self.mutateParams,
                program=Program1(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=ActionObject1(
                    action=a1, 
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)
            
            l2 = Learner1(
                initParams=self.mutateParams,
                program=Program1(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=ActionObject1(
                    action=a2, 
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = Team1(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):
                # select action
                act = random.choice(range(len(self.actionCodes)))

                # create new learner
                learner = Learner1(
                    initParams=self.mutateParams,
                    program=Program1(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    actionObj=ActionObject1(
                        action=act, 
                        initParams=self.mutateParams),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _scoreIndividuals(self, tasks, multiTaskType='min', doElites=True):
        # handle generation of new elites, typically just done in evolution
        if doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([team for team in self.rootTeams],
                                        key=lambda t: t.outcomes[task]))

        if len(tasks) == 1: # single fitness
            for team in self.rootTeams:
                team.fitness = team.outcomes[tasks[0]]
        else: # multi fitness
            # assign fitness to each agent based on tasks and score type
            if 'pareto' not in multiTaskType or 'lexicase' not in multiTaskType:
                self._simpleScorer(tasks, multiTaskType=multiTaskType)
            elif multiTaskType == 'paretoDominate':
                self._paretoDominateScorer(tasks)
            elif multiTaskType == 'paretoNonDominated':
                self._paretoNonDominatedScorer(tasks)
            elif multiTaskType == 'lexicaseStatic':
                self._lexicaseStaticScorer(tasks)
            elif multiTaskType == 'lexicaseDynamic':
                self._lexicaseDynamicScorer(tasks)

    def _simpleScorer(self, tasks, multiTaskType='min'):
        # first find min and max in each task
        mins = []
        maxs = []
        for task in tasks:
            mins.append(min([team.outcomes[task] for team in self.rootTeams]))
            maxs.append(max([team.outcomes[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                            for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    def _paretoDominateScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] >= t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness += 1

    def _paretoNonDominatedScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] < t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness -= 1

    def _lexicaseStaticScorer(self, tasks):
        stasks = list(tasks)
        random.shuffle(stasks)

        for rt in self.rootTeams:
            rt.fitness = rt.outcomes[tasks[0]]

    def _lexicaseDynamicScorer(self, tasks):
        pass

    def _saveFitnessStats(self):
        fitnesses = []
        for rt in self.rootTeams:
            fitnesses.append(rt.fitness)

        self.fitnessStats = {}
        self.fitnessStats['fitnesses'] = fitnesses
        self.fitnessStats['min'] = min(fitnesses)
        self.fitnessStats['max'] = max(fitnesses)
        self.fitnessStats['average'] = sum(fitnesses)/len(fitnesses)

    def getTaskStats(self, task):
        scores = []
        for rt in self.rootTeams:
            scores.append(rt.outcomes[task])

        scoreStats = {}
        scoreStats['scores'] = scores
        scoreStats['min'] = min(scores)
        scoreStats['max'] = max(scores)
        scoreStats['average'] = sum(scores)/len(scores)

        return scoreStats

    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness, reverse=True)
        numKeep = len(self.rootTeams) - int(len(self.rootTeams)*self.gap)
        deleteTeams = rankedTeams[numKeep:]

        for team in [t for t in deleteTeams if t not in self.elites]:
            # remove learners from team and delete team from populations
            if extraTeams is None or team not in extraTeams: team.removeLearners()
            self.teams.remove(team)
            self.rootTeams.remove(team)

        orphans = [learner for learner in self.learners if learner.numTeamsReferencing() == 0]
    
        for cursor in orphans:
            if not cursor.isActionAtomic(): # If the orphan does NOT point to an atomic action
                # Get the team the orphan is pointing to and remove the orphan's id from the team's in learner list
                cursor.actionObj.teamAction.inLearners.remove(str(cursor.id))

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0]
                
    def _generate(self, extraTeams=None):

        # extras who are already part of the team population
        protectedExtras = []
        extrasAdded = 0

        # add extras into the population
        if extraTeams is not None:
            for team in extraTeams:
                if team not in self.teams:
                    self.teams.append(team)
                    extrasAdded += 1
                else:
                    protectedExtras.append(team)

        oLearners = list(self.learners)
        oTeams = list(self.teams)

        # update generation in mutateParams
        self.mutateParams["generation"] = self.generation

        # get all the current root teams to be parents
        # mutate or clone
        while (len(self.teams) < self.teamPopSize + extrasAdded or
                (self.rootBasedPop and self.countRootTeams() < self.teamPopSize)):
            # get parent root team, and child to be based on that

            # ここをランダムではなく、階層上あるいは、過去の経験よりセレクトする。
            # rootTeamsを混ぜて、新しい、チームを作る。この時、そのチームは、プログラムへの１階層目のポインタを混ぜるだけである。
            parent = random.choice(self.rootTeams)
            child = Team1(initParams=self.mutateParams)

            # child starts just like parent
            for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                if new_learner.actionObj.teamAction is not None and new_learner.actionObj.teamAction in self.rootTeams:
                    referenced_rt = new_learner.actionObj.teamAction
                    clone = referenced_rt.clone()
                    self.teams.append(clone)

                    # new_learner's teamAction change to clone
                    new_learner.actionObj.teamAction = clone


                    # self.rootTeams.remove(rt)

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    def _nextEpoch(self):
        # add in newly added learners, and decide root teams
        self.rootTeams = []
        for team in self.teams:
            # add any new learners to the population
            for learner in team.learners:
                if learner not in self.learners:
                    #print("Adding {} to trainer learners".format(learner.id))
                    self.learners.append(learner)

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        self.generation += 1

    def removeHitchhikers(self, teams, visitedLearners):
        learnersRemoved = []
        teamsAffected = []

        for i, team in enumerate(teams):
            affected = False
            for learner in team.learners:
                # only remove if non atomic, or atomic and team has > 1 atomic actions
                if learner not in visitedLearners[i] and (
                        not learner.isActionAtomic() or 
                            (learner.isActionAtomic() and team.numAtomicActions() > 1)):
                    affected = True
                    learnersRemoved.append(learner)
                    team.removeLearner(learner)

            if affected:
                teamsAffected.append(team)

        return learnersRemoved, teamsAffected
    
    def validate_graph(self):
        print("Validating graph")

        print("Checking for broken learners")
        for cursor in self.learners:
            if cursor.isActionAtomic() and cursor.actionObj.actionCode == None:
                print("{} is an action atomic learner with no actionCode!".format(str(cursor.id)))
            if cursor.actionObj.teamAction == None and cursor.actionObj.actionCode == None:
                print("{} has no action!".format(str(cursor.id)))

        learner_map = {}
        for cursor in self.learners:
            if str(cursor.id) not in learner_map:
                learner_map[str(cursor.id)] = cursor.inTeams
            else:
                raise Exception("Duplicate learner id in trainer!")
        
        '''
        For every entry in the learner map check that the corresponding team has the learner 
        in its learners 
        '''
        for i,cursor in enumerate(learner_map.items()):
            for expected_team in cursor[1]:
                found_team = False
                for team in self.teams:
                    if str(team.id) == expected_team:
                        found_learner = 0
                        for learner in team.learners:
                            if str(learner.id) == cursor[0]:
                                found_learner += 1
                        if found_learner != 1:
                            print("found_learner = {} for learner {} in team {}".format(found_learner, cursor[0], str(team.id)))
                        found_team = True
                        break
                if found_team == False:
                    print("Could not find expected team {} in trainer".format(expected_team))
            print("learner {} inTeams valid [{}/{}]".format(cursor[0], i, len(learner_map.items())-1))

        '''
        Verify that for every inLearner in a team, the learner exists in the trainer, pointing to that team
        '''
        team_map = {}
        for cursor in self.teams:
            if str(cursor.id) not in team_map:
                team_map[str(cursor.id)] = cursor.inLearners
            else:
                raise Exception("Duplicate team id in trainer!")

        for i,cursor in enumerate(team_map.items()):
            for expected_learner in cursor[1]:
                found_learner = False
                points_to_team = False
                for learner in self.learners:
                    if str(learner.id) == expected_learner:
                        found_learner = True
                        if str(learner.actionObj.teamAction.id) == cursor[0]:
                            points_to_team = True
                            break
                if found_learner == False:
                    print("Could not find learner {} from team {} inLearners in trainer.".format(expected_learner, cursor[0]))
                if points_to_team == False:
                    print("Learner {} does not point to team {}".format(expected_learner, cursor[0]))
            print("team {} inLearners valid [{}/{}]".format(cursor[0], i, len(team_map.items())-1))
    
    def countRootTeams(self):
        numRTeams = 0
        for team in self.teams:
            if team.numLearnersReferencing() == 0: numRTeams += 1

        return numRTeams

    def get_graph(self):


        result = {
            "nodes":[],
            "links":[]
        }

        # First add action codes as nodes
        for actionCode in self.actionCodes:
            result["nodes"].append(
                {
                    "id": str(actionCode),   
                    "type": "action"
                }
            )
        
        # Then add teams as nodes
        for team in self.teams:
            result["nodes"].append(
                {
                    "id": str(team.id),
                    "type": "rootTeam" if team in self.rootTeams else "team"
                }
            )

        # Then add learners as nodes
        for learner in self.learners:
            result["nodes"].append(
                {
                    "id": str(learner.id),
                    "type": "learner"
                }
            )


        # Then add links from learners to teams
        for team in self.teams:
            for learner in team.inLearners:
                result["links"].append(
                    {
                        "source": learner,
                        "target": str(team.id)
                    }
                )
        
        # Then add links from teams to learners
        for learner in self.learners:
            for team in learner.inTeams:
                result["links"].append(
                    {
                        "source": team,
                        "target": str(learner.id)
                    }
                )
            
            # Also add links to action codes
            if learner.isActionAtomic():
                result["links"].append(
                    {
                        "source": str(learner.id),
                        "target": str(learner.actionObj.actionCode)
                    }
                )

        # with open("tpg_{}.json".format(self.generation), 'w') as out_file:
        #     json.dump(result, out_file)
        return result

    def cleanup(self):
        configurer._configureDefaults1(self, Trainer1, Agent1, Team1, Learner1, ActionObject1, Program1)

    def configFunctions(self):
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

    def saveToFile(self, fileName):
        pickle.dump(self, open(f'log/{fileName}.pickle', 'wb'))

class Trainer2:

    def __init__(self, 
        state=None, 
        teamPopSize:int=1000,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pMemMut:float=0.1,                  # *
        pMemAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nMemRegisters:int=4,
    ):
        if state is None: state = np.array([1.]*inputSize)

        '''
        Validate inputs
        '''
        int_greater_than_zero = {
            "teamPopSize": teamPopSize,
            "inputSize": inputSize,
            "nRegisters": nRegisters,
            "initMaxTeamSize": initMaxTeamSize,
            "initMaxProgSize": initMaxProgSize,
        }

        for entry in int_greater_than_zero.items(): self._must_be_integer_greater_than_zero(entry[0], entry[1])

        # Validate doElites
        if type(doElites) is not bool:  raise Exception("Invalid doElites")

        # Validate rootBasedPop
        if type(rootBasedPop) is not bool:  raise Exception("Invalid rootBasedPop")

        # Validate Traversal
        if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

        # Validate Probability parameters
        probabilities = {
            "pLrnDel": pLrnDel,
            "pLrnAdd": pLrnAdd,
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut,
            "pMemMut": pMemMut,
            "pMemAtom": pMemAtom,
            "pInstDel": pInstDel,
            "pInstAdd": pInstAdd,
            "pInstSwp": pInstSwp,
            "pInstMut": pInstMut
        }


        for entry in probabilities.items(): self._validate_probability(entry[0],entry[1])

        # Validate rampancy
        if (
            len(rampancy) != 3 or
            rampancy[0] < 0 or
            rampancy[2] < rampancy[1] or 
            rampancy[1] < 0 or
            rampancy[2] < 0
        ): raise Exception("Invalid rampancy parameter!", rampancy)

        # store all necessary params


        # population params
        self.teamPopSize = teamPopSize
        # whether population size is based on root teams or all teams
        self.rootBasedPop = rootBasedPop
        self.gap = gap # portion of root teams to remove each generation

        # input to agent (must be at-least size of input/state from environment)
        self.inputSize = inputSize # defaulted to number of Atari screen pixels
        # number of local memory registers each learner will have
        self.nRegisters = nRegisters

        # params for initializing evolution
        self.initMaxTeamSize = initMaxTeamSize # size of team = # of learners
        self.initMaxProgSize = initMaxProgSize # size of program = # of instructions

        # first store actions properly
        # self.doReal = self._setUpMemories(state)
        self.doReal = False
        # max team size possible throughout evolution
        self.maxTeamSize = maxTeamSize


        # params for continued evolution
        self.pLrnDel = pLrnDel
        self.pLrnAdd = pLrnAdd
        self.pLrnMut = pLrnMut
        self.pProgMut = pProgMut
        self.pMemMut = pMemMut
        self.pMemAtom = pMemAtom
        self.pInstDel = pInstDel
        self.pInstAdd = pInstAdd
        self.pInstSwp = pInstSwp
        self.pInstMut = pInstMut

        # whether to keep elites
        self.doElites = doElites

        if memType == "None":   self.memType = None
        self.memType = memType
        self.memMatrixShape = memMatrixShape

        self.rampancy = rampancy

        self.operationSet = operationSet

        self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        if self.doReal: nMemRegisters = max(max(self.memoryLengths), nMemRegisters)
        self.nMemRegisters = nMemRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {}
        self.actVars:       dict = {}
        self.functionsDict: dict = {}
        self.nOperations = None
        # configure tpg functions and variable appropriately now
        configurer.configure2(self, Trainer2, Agent2, Team2, Learner2, MemoryObject, Program2, self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)

    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[], task='task'):
        self.actVars['task']=task
        rTeams = [team for team in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in team.outcomes for task in skipTasks)]
        # breakpoint(random.sample(rTeams, len(rTeams)))
        for rt in rTeams:
            if not rt.outcomes.get(task): rt.outcomes[task]=0.

        if len(sortTasks) == 0: # just get all
            rootTeam = random.sample(rTeams, len(rTeams))
            return [Agent2(team, self.functionsDict, num=i, actVars=self.actVars)
                    for i,team in enumerate(rootTeam)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [Agent2(team, self.functionsDict, num=i, actVars=self.actVars)
                                for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]
            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [Agent2(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    def getEliteAgent(self, task):
        self.actVars['task']=task
        
        teams = [t for t in self.teams if task in t.outcomes]

        return Agent2(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                     self.functionsDict, num=0, actVars=self.actVars)

    def applyScores(self, scores): # used when multiprocessing
        for score in scores:
            for rt in self.rootTeams:
                if score[0] == rt.id:
                    for task, outcome in score[1].items():
                        rt.outcomes[task] = outcome
                    break # on to next score

        return self.rootTeams

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _states=None,_unexpectancies=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
            doElites=self.doElites
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams, _states=_states, _unexpectancies=_unexpectancies) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def _must_be_integer_greater_than_zero(self, name, value):
        if type(value) is not int or value <= 0:
            raise Exception(name + " must be integer greater than zero. Got " + str(value), name, value)

    def _validate_probability(self, name,  value):
        if type(value) is not float or value > 1.0 or value < 0.0:
            raise Exception(name + " is a probability, it must not be greater than 1.0 or less than 0.0", name, value)

    def _setUpMemories(self, state): #state: np.ndarray
        import numpy as np
        if not isinstance(state, np.ndarray): # list of lengths of each action
            raise Exception('state is not ndarray sequence')    # some may be real actions
        for _ in range(self.initMaxTeamSize):
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            MemoryObject.memories.append(key, state)
        doReal = False
        
        return doReal
    
    def resetMemories(self, state):
        import numpy as np
        assert isinstance(state, np.ndarray)
        for _ in range(self.initMaxTeamSize):
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            MemoryObject.memories.append(key, state)
        self.doReal = False
        
        if self.doReal: self.nMemRegisters = max(max(self.memoryLengths), self.nMemRegisters)
        # self.nActRegisters = self.nMemRegisters
        configurer.configure2(self, Trainer2, Agent2, Team2, Learner2, MemoryObject, Program2, 
            self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)
        self.initialize(_state=state)

    def initialize(self, _state=1):
        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            # m1,m2 = random.choices(range(len(self.memoryCodes)), 2)

            l1 = Learner2(
                initParams=self.mutateParams,
                program=Program2(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                memoryObj=MemoryObject(state=_state),
                numRegisters=self.nRegisters)
            
            l2 = Learner2(
                initParams=self.mutateParams,
                program=Program2(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                memoryObj=MemoryObject(state=_state),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = Team2(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):

                learner = Learner2(
                    initParams=self.mutateParams,
                    program=Program2(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    memoryObj=MemoryObject(state=_state),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _scoreIndividuals(self, tasks, multiTaskType='min', doElites=True):
        # handle generation of new elites, typically just done in evolution
        if doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(min([team for team in self.rootTeams], key=lambda t: t.outcomes[task]))

        if len(tasks) == 1: # single fitness
            for team in self.rootTeams:
                team.fitness = team.outcomes[tasks[0]]
        else: # multi fitness
            # assign fitness to each agent based on tasks and score type
            if 'pareto' not in multiTaskType or 'lexicase' not in multiTaskType:
                self._simpleScorer(tasks, multiTaskType=multiTaskType)
            elif multiTaskType == 'paretoDominate':
                self._paretoDominateScorer(tasks)
            elif multiTaskType == 'paretoNonDominated':
                self._paretoNonDominatedScorer(tasks)
            elif multiTaskType == 'lexicaseStatic':
                self._lexicaseStaticScorer(tasks)
            elif multiTaskType == 'lexicaseDynamic':
                self._lexicaseDynamicScorer(tasks)

    def _simpleScorer(self, tasks, multiTaskType='min'):
        # first find min and max in each task
        mins = []
        maxs = []
        for task in tasks:
            mins.append(min([team.outcomes[task] for team in self.rootTeams]))
            maxs.append(max([team.outcomes[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i]) for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i]) for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i]) for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    def _paretoDominateScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] >= t2.outcomes[task] for task in tasks]):
                    t1.fitness += 1

    def _paretoNonDominatedScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] < t2.outcomes[task] for task in tasks]):
                    t1.fitness -= 1

    def _lexicaseStaticScorer(self, tasks):
        stasks = list(tasks)
        random.shuffle(stasks)

        for rt in self.rootTeams:
            rt.fitness = rt.outcomes[tasks[0]]

    def _lexicaseDynamicScorer(self, tasks):
        pass

    def _saveFitnessStats(self):
        fitnesses = []
        for rt in self.rootTeams: fitnesses.append(rt.fitness)

        self.fitnessStats = {}
        self.fitnessStats['fitnesses'] = fitnesses
        self.fitnessStats['min'] = min(fitnesses)
        self.fitnessStats['max'] = max(fitnesses)
        self.fitnessStats['average'] = sum(fitnesses)/len(fitnesses)

    def getTaskStats(self, task):
        scores = []
        for rt in self.rootTeams:
            scores.append(rt.outcomes[task])

        scoreStats = {}
        scoreStats['scores'] = scores
        scoreStats['min'] = min(scores)
        scoreStats['max'] = max(scores)
        scoreStats['average'] = sum(scores)/len(scores)

        return scoreStats

    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness)
        numKeep = len(self.rootTeams) - int(len(self.rootTeams)*self.gap)
        deleteTeams = rankedTeams[numKeep:]

        for team in [t for t in deleteTeams if t not in self.elites]:
            # remove learners from team and delete team from populations
            if extraTeams is None or team not in extraTeams: team.removeLearners()
            self.teams.remove(team)
            self.rootTeams.remove(team)

        orphans = [learner for learner in self.learners if learner.numTeamsReferencing() == 0]

        for cursor in orphans:
            if not cursor.isMemoryAtomic(): # If the orphan does NOT point to an atomic action
                # Get the team the orphan is pointing to and remove the orphan's id from the team's in learner list
                cursor.memoryObj.teamMemory.inLearners.remove(str(cursor.id))
            else: # delete MemoryObject
                MemoryObject.memories.referenced[cursor.memoryObj.memoryCode]-=1

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0]

    def _generate(self, extraTeams=None, _states=None, _unexpectancies=None):

        # extras who are already part of the team population
        protectedExtras = []
        extrasAdded = 0

        # add extras into the population
        if extraTeams is not None:
            for team in extraTeams:
                if team not in self.teams:
                    self.teams.append(team)
                    extrasAdded += 1
                else:
                    protectedExtras.append(team)

        oLearners = list(self.learners)
        oTeams = list(self.teams)

        # update generation in mutateParams
        self.mutateParams["generation"] = self.generation
        # get all the current root teams to be parents
        # mutate or clone
        while (len(self.teams) < self.teamPopSize + extrasAdded or
                (self.rootBasedPop and self.countRootTeams() < self.teamPopSize)):
            # get parent root team, and child to be based on that
            # breakpoint('come here')

            # ここをランダムではなく、階層上あるいは、過去の経験よりセレクトする。
            # rootTeamsを混ぜて、新しい、チームを作る。この時、そのチームは、プログラムへの１階層目のポインタを混ぜるだけである。
            parent = random.choice(self.rootTeams)
            child = Team2(initParams=self.mutateParams)

            # child starts just like parent
            for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                if new_learner.memoryObj.teamMemory is not None and new_learner.memoryObj.teamMemory in self.rootTeams:
                    referenced_rt = new_learner.memoryObj.teamMemory
                    clone = referenced_rt.clone()
                    self.teams.append(clone)

                    # new_learner's teamAction change to clone
                    new_learner.memoryObj.teamMemory = clone

            if not _states is None and not _unexpectancies is None: 
                states = np.array(_states)
                unexpectancies = np.array(_unexpectancies)
                if len(unexpectancies[unexpectancies>0])!=0:
                    state = random.choices(states[unexpectancies>0], unexpectancies[unexpectancies>0])[0]
                else:
                    state = random.choices(states)[0]
                child.addLearner(Learner2(memoryObj=MemoryObject(state=state)))
                # breakpoint(state)
                # どういう時の状態をメモライズすれば良いか？
                # 理想は予想外に報酬の高い状態を記憶する。
                # 状態＋報酬　の　メモライズではどうだろうか？
                # つまり、報酬値の予想値との差異に基づいて、記憶すべき状態を優先づける。
                # この場合、
                # 

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    def _nextEpoch(self):
        # add in newly added learners, and decide root teams
        memory_code_list = set()
        self.rootTeams = []
        for team in self.teams:
            for learner in team.learners:
                if learner not in self.learners:
                    self.learners.append(learner)

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        for lrnr in self.learners:
            if lrnr.isMemoryAtomic():
                memory_code_list.add(lrnr.memoryObj.memoryCode)

        memory_code_list = list(memory_code_list)
        MemoryObject.memories.oblivion(memory_code_list)

        
        self.generation += 1

    def countRootTeams(self):
        numRTeams = 0
        for team in self.teams:
            if team.numLearnersReferencing() == 0: numRTeams += 1
        return numRTeams

    def cleanup(self):
        configurer._configureDefaults2(self, Trainer2, Agent2, Team2, Learner2, MemoryObject, Program2)

    def configFunctions(self):
        # first set up Agent functions
        Agent2.configFunctions(self.functionsDict["Agent"])

        # set up Team functions
        Team2.configFunctions(self.functionsDict["Team"])

        # set up Learner functions
        Learner2.configFunctions(self.functionsDict["Learner"])

        # set up Program functions
        Program2.configFunctions(self.functionsDict["Program"])

    def saveToFile(self, fileName):
        pickle.dump(self, open(f'log/{fileName}.pickle', 'wb'))
        MemoryObject.backup(fileName=fileName)
    
class Trainer3:

    def __init__(self, 
        actions=2, 
        teamPopSize:int=1000,               # *
        rootBasedPop:bool=True,             
        gap:float=0.5,                      
        inputSize:int=33600,                
        nRegisters:int=8,                   # *
        initMaxTeamSize:int=10,             # *
        initMaxProgSize:int=10,             # *
        maxTeamSize:int=-1,                 # *
        pLrnDel:float=0.7,                  # *
        pLrnAdd:float=0.6,                  # *
        pLrnMut:float=0.2,                  # *
        pProgMut:float=0.1,                 # *
        pActMut:float=0.1,                  # *
        pActAtom:float=0.95,                # *
        pInstDel:float=0.5,                 # *
        pInstAdd:float=0.4,                 # *
        pInstSwp:float=0.2,                 # *
        pInstMut:float=1.0,                 # *
        doElites:bool=True, 
        memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        operationSet:str="custom", 
        traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nActRegisters:int=4
    ):


        '''
        Validate inputs
        '''
        int_greater_than_zero = {
            "teamPopSize": teamPopSize,
            "inputSize": inputSize,
            "nRegisters": nRegisters,
            "initMaxTeamSize": initMaxTeamSize,
            "initMaxProgSize": initMaxProgSize,
        }

        for entry in int_greater_than_zero.items(): self._must_be_integer_greater_than_zero(entry[0], entry[1])

        # Validate doElites
        if type(doElites) is not bool:  raise Exception("Invalid doElites")

        # Validate rootBasedPop
        if type(rootBasedPop) is not bool:  raise Exception("Invalid rootBasedPop")

        # Validate Traversal
        if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

        # Validate Probability parameters
        probabilities = {
            "pLrnDel": pLrnDel,
            "pLrnAdd": pLrnAdd,
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut,
            "pActMut": pActMut,
            "pActAtom": pActAtom,
            "pInstDel": pInstDel,
            "pInstAdd": pInstAdd,
            "pInstSwp": pInstSwp,
            "pInstMut": pInstMut
        }


        for entry in probabilities.items(): self._validate_probability(entry[0],entry[1])

        # Validate rampancy
        if (
            len(rampancy) != 3 or
            rampancy[0] < 0 or
            rampancy[2] < rampancy[1] or 
            rampancy[1] < 0 or
            rampancy[2] < 0
        ): raise Exception("Invalid rampancy parameter!", rampancy)

        # store all necessary params

        # first store actions properly
        self.doReal = False #self._setUpActions(actions)

        # population params
        self.teamPopSize = teamPopSize
        # whether population size is based on root teams or all teams
        self.rootBasedPop = rootBasedPop
        self.gap = gap # portion of root teams to remove each generation

        # input to agent (must be at-least size of input/state from environment)
        self.inputSize = inputSize # defaulted to number of Atari screen pixels
        # number of local memory registers each learner will have
        self.nRegisters = nRegisters

        # params for initializing evolution
        self.initMaxTeamSize = initMaxTeamSize # size of team = # of learners
        self.initMaxProgSize = initMaxProgSize # size of program = # of instructions

        # max team size possible throughout evolution
        self.maxTeamSize = maxTeamSize

        # params for continued evolution
        self.pLrnDel = pLrnDel
        self.pLrnAdd = pLrnAdd
        self.pLrnMut = pLrnMut
        self.pProgMut = pProgMut
        self.pActMut = pActMut
        self.pActAtom = pActAtom
        self.pInstDel = pInstDel
        self.pInstAdd = pInstAdd
        self.pInstSwp = pInstSwp
        self.pInstMut = pInstMut

        # whether to keep elites
        self.doElites = doElites

        if memType == "None":   self.memType = None
        self.memType = memType
        self.memMatrixShape = memMatrixShape

        self.rampancy = rampancy

        self.operationSet = operationSet

        self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nActRegisters = nActRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {}
        self.actVars:       dict = {}
        self.functionsDict: dict = {}
        self.nOperations = None



        # configure tpg functions and variable appropriately now
        configurer.configure3(self, Trainer3, Agent3, Team3, Learner3, ActionObject3, Program3, self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)

    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[], task='task'):
        self.actVars['task']=task
        # remove those that get skipped
        rTeams = [team for team in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in team.outcomes for task in skipTasks)]
        for rt in rTeams:
            if not rt.outcomes.get(task): rt.outcomes[task]=0.

        if len(sortTasks) == 0: # just get all
            rootTeam = random.sample(rTeams, len(rTeams))
            return [Agent3(team, self.functionsDict, num=i, actVars=self.actVars)
                    for i,team in enumerate(rootTeam)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [Agent3(team, self.functionsDict, num=i, actVars=self.actVars)
                                for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]
            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [Agent3(team, self.functionsDict, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    def getEliteAgent(self, task):
        
        teams = [t for t in self.teams if task in t.outcomes]

        return Agent3(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                     self.functionsDict, num=0, actVars=self.actVars)

    def applyScores(self, scores): # used when multiprocessing
        for score in scores:
            for rt in self.rootTeams:
                if score[0] == rt.id:
                    for task, outcome in score[1].items():
                        rt.outcomes[task] = outcome
                    break # on to next score

        return self.rootTeams

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
            doElites=self.doElites
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def _must_be_integer_greater_than_zero(self, name, value):
        if type(value) is not int or value <= 0:
            raise Exception(name + " must be integer greater than zero. Got " + str(value), name, value)

    def _validate_probability(self, name,  value):
        if type(value) is not float or value > 1.0 or value < 0.0:
            raise Exception(name + " is a probability, it must not be greater than 1.0 or less than 0.0", name, value)

    def _setUpActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes:list = range(actions)
            ActionObject3.actions=self.actionCodes

            doReal = False
        else: # list of lengths of each action
            # some may be real actions
            self.actionCodes:list = range(len(actions))
            self.actionLengths = list(actions)
            doReal = True
            
        return doReal
    
    def resetActions(self, actions):
        if isinstance(actions, int):
            # all discrete actions
            self.actionCodes:list = range(actions)
            self.doReal = False
            ActionObject3.actions=self.actionCodes
        else: # list of lengths of each action
            # some may be real actions
            self.actionCodes:list = range(len(actions))
            ActionObject3.actions=self.actionCodes
            self.actionLengths = list(actions)
            self.doReal = True
        
        if self.doReal: self.nActRegisters = max(max(self.actionLengths), self.nActRegisters)
        self.nActRegisters = self.nActRegisters
        configurer.configure3(self, Trainer3, Agent3, Team3, Learner3, ActionObject3, Program3, 
            self.memType is not None, self.memType, self.doReal, self.operationSet, self.traversal)
        self.initialize()

    def initialize(self):
        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            a1,a2 = random.sample(range(len(ActionObject3.actions)), 2)

            l1 = Learner3(
                initParams=self.mutateParams,
                program=Program3(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=ActionObject3(
                    action=a1, 
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)
            
            l2 = Learner3(
                initParams=self.mutateParams,
                program=Program3(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=ActionObject3(
                    action=a2, 
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = Team3(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):
                # select action
                act = random.choice(range(len(self.actionCodes)))

                # create new learner
                learner = Learner3(
                    initParams=self.mutateParams,
                    program=Program3(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    actionObj=ActionObject3(
                        action=act, 
                        initParams=self.mutateParams),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _scoreIndividuals(self, tasks, multiTaskType='min', doElites=True):
        # handle generation of new elites, typically just done in evolution
        if doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([team for team in self.rootTeams],
                                        key=lambda t: t.outcomes[task]))

        if len(tasks) == 1: # single fitness
            for team in self.rootTeams:
                team.fitness = team.outcomes[tasks[0]]
        else: # multi fitness
            # assign fitness to each agent based on tasks and score type
            if 'pareto' not in multiTaskType or 'lexicase' not in multiTaskType:
                self._simpleScorer(tasks, multiTaskType=multiTaskType)
            elif multiTaskType == 'paretoDominate':
                self._paretoDominateScorer(tasks)
            elif multiTaskType == 'paretoNonDominated':
                self._paretoNonDominatedScorer(tasks)
            elif multiTaskType == 'lexicaseStatic':
                self._lexicaseStaticScorer(tasks)
            elif multiTaskType == 'lexicaseDynamic':
                self._lexicaseDynamicScorer(tasks)

    def _simpleScorer(self, tasks, multiTaskType='min'):
        # first find min and max in each task
        mins = []
        maxs = []
        for task in tasks:
            mins.append(min([team.outcomes[task] for team in self.rootTeams]))
            maxs.append(max([team.outcomes[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt.outcomes[task]-mins[i])/(maxs[i]-mins[i])
                            for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    def _paretoDominateScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] >= t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness += 1

    def _paretoNonDominatedScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1.outcomes[task] < t2.outcomes[task]
                         for task in tasks]):
                    t1.fitness -= 1

    def _lexicaseStaticScorer(self, tasks):
        stasks = list(tasks)
        random.shuffle(stasks)

        for rt in self.rootTeams:
            rt.fitness = rt.outcomes[tasks[0]]

    def _lexicaseDynamicScorer(self, tasks):
        pass

    def _saveFitnessStats(self):
        fitnesses = []
        for rt in self.rootTeams:
            fitnesses.append(rt.fitness)

        self.fitnessStats = {}
        self.fitnessStats['fitnesses'] = fitnesses
        self.fitnessStats['min'] = min(fitnesses)
        self.fitnessStats['max'] = max(fitnesses)
        self.fitnessStats['average'] = sum(fitnesses)/len(fitnesses)

    def getTaskStats(self, task):
        scores = []
        for rt in self.rootTeams:
            scores.append(rt.outcomes[task])

        scoreStats = {}
        scoreStats['scores'] = scores
        scoreStats['min'] = min(scores)
        scoreStats['max'] = max(scores)
        scoreStats['average'] = sum(scores)/len(scores)

        return scoreStats

    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness, reverse=True)
        numKeep = len(self.rootTeams) - int(len(self.rootTeams)*self.gap)
        deleteTeams = rankedTeams[numKeep:]

        for team in [t for t in deleteTeams if t not in self.elites]:
            # remove learners from team and delete team from populations
            if extraTeams is None or team not in extraTeams: team.removeLearners()
            self.teams.remove(team)
            self.rootTeams.remove(team)

        orphans = [learner for learner in self.learners if learner.numTeamsReferencing() == 0]
    
        for cursor in orphans:
            if not cursor.isActionAtomic(): # If the orphan does NOT point to an atomic action
                # Get the team the orphan is pointing to and remove the orphan's id from the team's in learner list
                cursor.actionObj.teamAction.inLearners.remove(str(cursor.id))

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0]
                
    def _generate(self, extraTeams=None):

        # extras who are already part of the team population
        protectedExtras = []
        extrasAdded = 0

        # add extras into the population
        if extraTeams is not None:
            for team in extraTeams:
                if team not in self.teams:
                    self.teams.append(team)
                    extrasAdded += 1
                else:
                    protectedExtras.append(team)

        oLearners = list(self.learners)
        oTeams = list(self.teams)

        # update generation in mutateParams
        self.mutateParams["generation"] = self.generation

        # get all the current root teams to be parents
        # mutate or clone
        while (len(self.teams) < self.teamPopSize + extrasAdded or
                (self.rootBasedPop and self.countRootTeams() < self.teamPopSize)):
            # get parent root team, and child to be based on that

            # ここをランダムではなく、階層上あるいは、過去の経験よりセレクトする。
            # rootTeamsを混ぜて、新しい、チームを作る。この時、そのチームは、プログラムへの１階層目のポインタを混ぜるだけである。
            parent = random.choice(self.rootTeams)
            child = Team3(initParams=self.mutateParams)

            # child starts just like parent
            for learner in parent.learners: child.addLearner(learner)

            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                if new_learner.actionObj.teamAction is not None and new_learner.actionObj.teamAction in self.rootTeams:
                    referenced_rt = new_learner.actionObj.teamAction
                    clone = referenced_rt.clone()
                    self.teams.append(clone)

                    new_learner.actionObj.teamAction = clone

            self.teams.append(child)

        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    def _nextEpoch(self):
        # add in newly added learners, and decide root teams
        self.rootTeams = []
        for team in self.teams:
            # add any new learners to the population
            assert isinstance(team, Team3)
            for learner in team.learners:
                if learner not in self.learners:
                    self.learners.append(learner)

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        self.generation += 1

    def countRootTeams(self):
        numRTeams = 0
        for team in self.teams:
            if team.numLearnersReferencing() == 0: numRTeams += 1

        return numRTeams

    def cleanup(self):
        configurer._configureDefaults3(self, Trainer3, Agent3, Team3, Learner3, ActionObject3, Program3)

    def configFunctions(self):
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

    def saveToFile(self, fileName):
        pickle.dump(self, open(f'log/{fileName}.pickle', 'wb'))
