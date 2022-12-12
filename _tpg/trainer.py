from _tpg.utils import *
from _tpg.utils import _Logger
import random
import uuid
import pickle
import numpy as np


class _Trainer(_Logger):
    Agent = None
    Team = None
    Learner = None
    Program = None
    ActionObject = None
    MemoryObject = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
            from _tpg.agent import _Agent
            from _tpg.team import _Team
            from _tpg.learner import _Learner
            from _tpg.program import _Program
            from _tpg.action_object import _ActionObject

            cls.Agent = _Agent
            cls.Team = _Team
            cls.Learner = _Learner
            cls.Program = _Program
            cls.ActionObject = _ActionObject

        return super().__new__(cls, *args, **kwargs)        

    def __init__(self, 
        actions=None, 
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
        # memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        # operationSet:str="custom", 
        # traversal:str="team", 
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
        # if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        # if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

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


        # whether to keep elites
        self.doElites = doElites

        # if memType == "None":   self.memType = None
        # self.memType = memType
        # self.memMatrixShape = memMatrixShape
        self.memMatrix = np.zeros(shape=memMatrixShape)

        self.rampancy = rampancy

        # self.operationSet = operationSet

        # self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        # if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nActRegisters = nActRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this
        self.nOperations = 16

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {
            "generation": self.generation,
            "maxTeamSize": maxTeamSize,
            "pLrnAdd": pLrnAdd, 
            "pLrnDel": pLrnDel, 
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut, 
            "pActAtom": pActAtom, 
            "pActMut": pActMut, 
            "pInstAdd": pInstAdd, 
            "pInstDel": pInstDel, 
            "pInstMut": pInstMut,
            "pInstSwp": pInstSwp, 
            "nOperations": self.nOperations,
            "nDestinations": nRegisters,
            "inputSize": inputSize, 
            "initMaxProgSize": initMaxProgSize,
            "rampantGen": rampancy[0], 
            "rampantMin": rampancy[1], 
            "rampantMax": rampancy[2], 
            "idCountTeam": 0, 
            "idCountLearner": 0, 
            "idCountProgram": 0
        }
        self.actVars:       dict = {
            "frameNum": 0,
            "task": [],
            "memMatrix": self.memMatrix,
        }
        # self.logger=None

        if actions: self.setActions(actions)

    def _must_be_integer_greater_than_zero(self, name, value):
        if type(value) is not int or value <= 0:
            raise Exception(name + " must be integer greater than zero. Got " + str(value), name, value)

    def _validate_probability(self, name,  value):
        if type(value) is not float or value > 1.0 or value < 0.0:
            raise Exception(name + " is a probability, it must not be greater than 1.0 or less than 0.0", name, value)

    def _initialize(self):

        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            a1,a2 = random.sample(self.__class__.ActionObject.actions, 2)

            l1 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=self.__class__.ActionObject(action=a1),
                numRegisters=self.nRegisters)
            
            l2 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=self.__class__.ActionObject(action=a2),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = self.__class__.Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):
                # select action
                act = random.choice(self.__class__.ActionObject.actions)

                # create new learner
                learner = self.__class__.Learner(
                    initParams=self.mutateParams,
                    program=self.__class__.Program(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    actionObj=self.__class__.ActionObject(
                        action=act),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _scoreIndividuals(self, tasks, multiTaskType='min'):
        # handle generation of new elites, typically just done in evolution
        assert isinstance(tasks, list), f'{tasks} is not list'

        if self.doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([rt for rt in self.rootTeams],
                                        key=lambda t: t[task]))

        if len(tasks) == 1: # single fitness
            for rt in self.rootTeams:
                rt.fitness = rt[tasks[0]]
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
            mins.append(min([team[task] for team in self.rootTeams]))
            maxs.append(max([team[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt[task]-mins[i])/(maxs[i]-mins[i]+0.001)
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt[task]-mins[i])/(maxs[i]-mins[i]+0.001)
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt[task]-mins[i])/(maxs[i]-mins[i]+0.001)
                            for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    def _paretoDominateScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1[task] >= t2[task]
                         for task in tasks]):
                    t1.fitness += 1

    def _paretoNonDominatedScorer(self, tasks):
        for t1 in self.rootTeams:
            t1.fitness = 0
            for t2 in self.rootTeams:
                if t1 == t2:
                    continue # don't compare to self

                # compare on all tasks
                if all([t1[task] < t2[task]
                         for task in tasks]):
                    t1.fitness -= 1

    def _lexicaseStaticScorer(self, tasks):
        stasks = list(tasks)
        random.shuffle(stasks)

        for rt in self.rootTeams:
            rt.fitness = rt[tasks[0]]

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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

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
  
    def countRootTeams(self):
        numRTeams = 0
        for team in self.teams:
            if team.numLearnersReferencing() == 0: numRTeams += 1

        return numRTeams

    def setActions(self, actions):
        self.__class__.ActionObject.actions = range(actions)
        self._initialize()
        return self.__class__.ActionObject.actions

    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[], task='task'):
        self.actVars['task']=task
        # remove those that get skipped
        rTeams = [rt for rt in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in rt.outcomes for task in skipTasks)]

        if len(sortTasks) == 0: # just get all
            return [self.__class__.Agent(team, num=i, actVars=self.actVars)
                    for i,team in enumerate(rTeams)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [self.__class__.Agent(team, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]

            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [self.__class__.Agent(team, num=i, actVars=self.actVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    def getEliteAgent(self, task):
        
        teams = [t for t in self.teams if task in t.outcomes]

        return self.__class__.Agent(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                        num=0, actVars=self.actVars)

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

    def applyScores(self, scores): # used when multiprocessing
        for score in scores:
            for rt in self.rootTeams:
                if score[0] == rt.id:
                    for task, outcome in score[1].items():
                        rt.outcomes[task] = outcome
                    break # on to next score

        return self.rootTeams

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        assert len(self.rootTeams) != 0, 'root teams is null'
        # self.info(f'evolving:{datetime.now()}, tasks:{tasks}')
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def save(self, fileName):
        self._actions = self.__class__.ActionObject.actions
        with open(f'{fileName}.pickle', 'wb') as _trainer:
            pickle.dump(self, _trainer)

    @classmethod
    def load(cls, fileName:str):
        trainer = None
        with open(f'{fileName}.pickle', 'rb') as _trainer:
            trainer = pickle.load(_trainer)
            assert isinstance(trainer, cls), f'this file is not {cls}'
        
        trainer.ActionObject.actions = trainer._actions
        return trainer

    # @property
    def getElite(self, tasks):
        self._scoreIndividuals(tasks)
        return self.__class__.Agent(max([rt for rt in self.rootTeams],
                        key=lambda rt: rt.fitness),
                        num=0, actVars=self.actVars)

class Trainer(_Trainer):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import _Agent
            from _tpg.team import _Team
            from _tpg.learner import Learner
            from _tpg.program import Program
            from _tpg.action_object import _ActionObject

            cls._instance = True
            cls.Agent = _Agent
            cls.Team = _Team
            cls.Learner = Learner
            cls.Program = Program
            cls.ActionObject = _ActionObject

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        actions=None, 
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
        # memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        # operationSet:str="custom", 
        # traversal:str="team", 
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
        # if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        # if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

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


        # whether to keep elites
        self.doElites = doElites

        # if memType == "None":   self.memType = None
        # self.memType = memType
        # self.memMatrixShape = memMatrixShape
        self.memMatrix = np.zeros(shape=memMatrixShape)

        self.rampancy = rampancy

        # self.operationSet = operationSet

        # self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        # if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nActRegisters = nActRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this
        self.nOperations = 18

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {
            "generation": self.generation,
            "maxTeamSize": maxTeamSize,
            "pLrnAdd": pLrnAdd, 
            "pLrnDel": pLrnDel, 
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut, 
            "pActAtom": pActAtom, 
            "pActMut": pActMut, 
            "pInstAdd": pInstAdd, 
            "pInstDel": pInstDel, 
            "pInstMut": pInstMut,
            "pInstSwp": pInstSwp, 
            "nOperations": self.nOperations,
            "nDestinations": nRegisters,
            "inputSize": inputSize, 
            "initMaxProgSize": initMaxProgSize,
            "rampantGen": rampancy[0], 
            "rampantMin": rampancy[1], 
            "rampantMax": rampancy[2], 
            "idCountTeam": 0, 
            "idCountLearner": 0, 
            "idCountProgram": 0
        }
        self.actVars:       dict = {
            "frameNum": 0,
            "task": [],
            "memMatrix": self.memMatrix,
        }
        # self.logger=None

        if actions: self.setActions(actions)

class Trainer1(Trainer):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1
            from _tpg.team import Team1
            from _tpg.learner import Learner1
            from _tpg.program import Program1
            from _tpg.action_object import _ActionObject

            cls._instance = True
            cls.Agent = Agent1
            cls.Team = Team1
            cls.Learner = Learner1
            cls.Program = Program1
            cls.ActionObject= _ActionObject

        return super().__new__(cls, *args, **kwargs)
    
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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getActionTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

class Trainer1_1(Trainer1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1
            from _tpg.team import Team1_1
            from _tpg.learner import Learner1_1
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject1

            cls._instance = True
            cls.Agent = Agent1
            cls.Team = Team1_1
            cls.Learner = Learner1_1
            cls.Program = Program1
            cls.ActionObject = ActionObject1
            # cls.ActionObject._nan = cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

    def _initialize(self):

        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            a1,a2 = self.__class__.ActionObject.actions.choices(k=2)

            l1 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=self.__class__.ActionObject(action=a1),
                numRegisters=self.nRegisters)
            
            l2 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                actionObj=self.__class__.ActionObject(action=a2),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = self.__class__.Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):
                # select action
                act = self.__class__.ActionObject.actions.choice()

                # create new learner
                learner = self.__class__.Learner(
                    initParams=self.mutateParams,
                    program=self.__class__.Program(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    actionObj=self.__class__.ActionObject(action=act),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _generate(self, extraTeams=None, _actionSequence=None, _actionReward=None):
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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getActionTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            if not _actionSequence is None and not _actionReward is None: 
                # assert not isinstance(_actionSequence, dict) and not isinstance(_actionReward, dict), f'{_actionSequence}, {_actionReward}'
                sequences = np.array(_actionSequence)
                rewards = np.array(_actionReward)
                assert len(sequences) == len(rewards), f'{sequences.shape}:{rewards.shape}'
                rewards = sigmoid(rewards)*1000
                # randome choice story by rewards
                sequence = random.choices(sequences, rewards)[0]
                actionCode = self.__class__.ActionObject.actions.append(sequence)
                child.addLearner(self.__class__.Learner(actionObj=self.__class__.ActionObject(action=actionCode)))
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
        action_code_list = set()
        self.rootTeams = []
        for team in self.teams:
            for learner in team.learners:
                if learner not in self.learners:
                    self.learners.append(learner)

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                action_code_list.add(lrnr.actionObj.actionCode)

        action_code_list = list(action_code_list)
        self.__class__.ActionObject.actions.oblivion(action_code_list)

        self.generation += 1

    def setActions(self, actions):
        for i in range(actions):
            self.__class__.ActionObject.actions.append([i])
        self._initialize()
        return self.__class__.ActionObject.actions

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _actionSequence=None, _actionReward=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams, _actionSequence=_actionSequence, _actionReward=_actionReward) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

class Trainer1_2(Trainer1_1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_1
            from _tpg.team import Team1_2
            from _tpg.learner import Learner1_2
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject2

            cls._instance = True
            cls.Agent = Agent1_1
            cls.Team = Team1_2
            cls.Learner = Learner1_2
            cls.Program = Program1
            cls.ActionObject = ActionObject2
            # cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)
        
    def setActions(self, actions):
        for i in range(actions):
            self.__class__.ActionObject.actions.append([int(i)])
        self._initialize()
        return self.__class__.ActionObject.actions

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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getActionTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)


    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch()

class Trainer1_2_1(Trainer1_2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_2
            from _tpg.team import Team1_2_1
            from _tpg.learner import Learner1_2_1
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject3

            cls._instance = True
            cls.Agent = Agent1_2
            cls.Team = Team1_2_1
            cls.Learner = Learner1_2_1
            cls.Program = Program1
            cls.ActionObject = ActionObject3
            cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

    def _initialize(self):

        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            l1 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)
            
            l2 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = self.__class__.Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):
                # select action
                # act = self.__class__.ActionObject.actions.choice()

                # create new learner
                learner = self.__class__.Learner(
                    initParams=self.mutateParams,
                    program=self.__class__.Program(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _select(self, extraTeams=None):

        # Teams remove
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
                # Get the team the orphan is pointing to and remove the orphan's  _id from the team's in learner list
                assert cursor._id in cursor.actionObj.teamAction.inLearners, f'{cursor._id} not in {cursor.actionObj.teamAction.inLearners}'
                cursor.actionObj.teamAction.inLearners.remove(cursor._id)
            
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

            parent = random.choice(self.rootTeams)
            parent.addSequence()
            child = parent.clone

            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)


            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getActionTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    def _nextEpoch(self):
        # add in newly added learners, and dec _ide root teams
        self.rootTeams = []
        for team in self.teams:
            # add any new learners to the population
            # team.extinction*=1.01
            assert len(team.inLearners)==0 or any(isinstance(i, uuid.UUID) for i in team.inLearners), f'must be uuid in {team.inLearners}, {[i for i in team.inLearners]}'
            
            for learner in team.learners:
                if learner not in self.learners:
                    #print("Adding {} to trainer learners".format(learner. _id))
                    self.learners.append(learner)

            # self.debug(f'team_sequences:{team.sequence}')
            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)


        action_code_list = set()
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                action_code_list.add(lrnr.actionObj.actionCode)

        action_code_list = list(action_code_list)
        self.__class__.ActionObject.actions.oblivion(action_code_list)

        self.generation += 1

    def setActions(self, actions):
        if self.__class__.ActionObject.actions:
            for i in range(actions):
                self.__class__.ActionObject.actions.append([i], _weight=0.)
        else:
            breakpoint('notImpletemted')
        self._initialize()
        return self.__class__.ActionObject.actions

class Trainer1_2_2(Trainer1_2_1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_2
            from _tpg.team import Team1_2_2
            from _tpg.learner import Learner1_2_2
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject4

            cls._instance = True
            cls.Agent = Agent1_2
            cls.Team = Team1_2_2
            cls.Learner = Learner1_2_2
            cls.Program = Program1
            cls.ActionObject = ActionObject4
            cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

class Trainer1_2_3(Trainer1_2_2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_2_1
            from _tpg.team import Team1_2_3
            from _tpg.learner import Learner1_2_3
            from _tpg.program import Program1
            from _tpg.memory_object import ActionObject5

            cls._instance = True
            cls.Agent = Agent1_2_1
            cls.Team = Team1_2_3
            cls.Learner = Learner1_2_3
            cls.Program = Program1
            cls.ActionObject = ActionObject5
            cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

    def _scoreIndividuals(self, tasks, multiTaskType='min'):
        # handle generation of new elites, typically just done in evolution
        assert isinstance(tasks, list), f'{tasks} is not list'

        if self.doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([rt for rt in self.rootTeams],
                                        key=lambda t: t[task]))

        if len(tasks) == 1: # single fitness
            for rt in self.rootTeams:
                rt.fitness = rt[tasks[0]]
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

        for t in self.teams:
            t.allocation()

    def _select(self, extraTeams=None):

        # rankedLearners = sorted(self.learners, key=lambda l: l.fitness, reverse=True)
        # nK = len(self.learners) - int(len(self.learners)*self.gap)
        # deleteLearners_candidate = rankedLearners[nK:]
        # deleteLearners_candidate = list(filter(lambda l: (l.extinction >1) or (l in deleteLearners_candidate), self.learners))

        deleteLearners = [l for l in self.learners if (l.isActionAtomic() and l.actionObj.weight>0.9**12) ]
        
        for t in self.teams:
            for dl in deleteLearners:
                if dl in t.learners:
                    t.removeLearner(dl)

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness, reverse=True)
        numKeep = len(self.rootTeams) - int(len(self.rootTeams)*self.gap)
        deleteTeams = rankedTeams[numKeep:]

        for team in [t for t in deleteTeams if t not in self.elites]:
            # remove learners from team and delete team from populations
            if extraTeams is None or team not in extraTeams: team.removeLearners()
            self.teams.remove(team)
            self.rootTeams.remove(team)

        orphans = [learner for learner in self.learners if learner.numTeamsReferencing() == 0]
    
        for cursor in orphans+deleteLearners:
            if not cursor.isActionAtomic() \
                and cursor._id in cursor.actionObj.teamAction.inLearners:
                #  f'{cursor._id} not in {cursor.actionObj.teamAction.inLearners}'
                cursor.actionObj.teamAction.inLearners.remove(cursor._id)

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0 or not learner in deleteLearners]
        self.debug(f'learners_len:{len(self.learners)}, actions_len:{len(self.__class__.ActionObject.actions)}')

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

            parent = random.choice(self.rootTeams)
            parent.addSequence()
            child = parent.clone

            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)


            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getActionTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            self.teams.append(child)


        # remove unused extras
        if extraTeams is not None:
            for team in extraTeams:
                if team.numLearnersReferencing() == 0 and team not in protectedExtras:
                    self.teams.remove(team)

    def _nextEpoch(self):
        # add in newly added learners, and dec _ide root teams
        self.rootTeams = []
        for team in self.teams:
            # add any new learners to the population
            # team.extinction*=1.01
            assert len(team.inLearners)==0 or any(isinstance(i, uuid.UUID) for i in team.inLearners), f'must be uuid in {team.inLearners}, {[i for i in team.inLearners]}'
            
            for learner in team.learners:
                if learner not in self.learners:
                    #print("Adding {} to trainer learners".format(learner. _id))
                    self.learners.append(learner)

            # self.debug(f'team_sequences:{team.sequence}')
            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)


        action_code_list = set()
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                action_code_list.add(lrnr.actionObj.actionCode)

        action_code_list = list(action_code_list)
        self.__class__.ActionObject.actions.oblivion(action_code_list)

        self.generation += 1

class Trainer1_3(Trainer1_2):
    Hippocampus=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_3
            from _tpg.team import Team1_3
            from _tpg.learner import Learner1_3
            from _tpg.program import Program1_3
            from _tpg.memory_object import Qualia

            cls._instance = True
            cls.Agent = Agent1_3
            cls.Team = Team1_3
            cls.Learner = Learner1_3
            cls.Program = Program1_3
            cls.Hippocampus = Trainer2_3
            cls.ActionObject = Qualia
            cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

class Trainer1_3_1(Trainer1_3):
    Hippocampus=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_3
            from _tpg.team import Team1_3_1
            from _tpg.learner import Learner1_3_1
            from _tpg.program import Program1_3
            from _tpg.memory_object import Qualia

            cls._instance = True
            cls.Agent = Agent1_3
            cls.Team = Team1_3_1
            cls.Learner = Learner1_3_1
            cls.Program = Program1_3
            cls.Hippocampus = Trainer2_3
            cls.ActionObject = Qualia
            cls.ActionObject._nan = cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

class Trainer1_3_2(Trainer1_3):
    Hippocampus=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent1_3
            from _tpg.team import Team1_3_2
            from _tpg.learner import Learner1_3_2
            from _tpg.program import Program1_3
            from _tpg.memory_object import Operation

            cls._instance = True
            cls.Agent = Agent1_3
            cls.Team = Team1_3_2
            cls.Learner = Learner1_3_2
            cls.Program = Program1_3
            cls.Hippocampus = Trainer2_3
            cls.ActionObject = Operation
            cls.ActionObject._nan = cls.ActionObject()

        return super().__new__(cls, *args, **kwargs)

class Trainer2(Trainer):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent2
            from _tpg.team import Team2
            from _tpg.learner import Learner2
            from _tpg.program import Program2
            # from _tpg.action_object import _ActionObject
            from _tpg.memory_object import _MemoryObject

            cls._instance = True
            cls.Agent = Agent2
            cls.Team = Team2
            cls.Learner = Learner2
            cls.Program = Program2
            # cls.ActionObject = _ActionObject
            cls.MemoryObject = _MemoryObject
            cls.MemoryObject()

        return super().__new__(cls, *args, **kwargs)

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
        # memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        # operationSet:str="custom", 
        # traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nMemRegisters:int=4
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
        # if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        # if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

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

        # first store actions properly
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


        # whether to keep elites
        self.doElites = doElites

        # if memType == "None":   self.memType = None
        # self.memType = memType
        # self.memMatrixShape = memMatrixShape
        self.memMatrix = np.zeros(shape=memMatrixShape)

        self.rampancy = rampancy

        # self.operationSet = operationSet

        # self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        # if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nMemRegisters = nMemRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {
            "generation": self.generation,
            "maxTeamSize": maxTeamSize,
            "pLrnAdd": pLrnAdd, 
            "pLrnDel": pLrnDel, 
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut, 
            "pMemAtom": pMemAtom, 
            "pMemMut": pMemMut, 
            "pInstAdd": pInstAdd, 
            "pInstDel": pInstDel, 
            "pInstMut": pInstMut,
            "pInstSwp": pInstSwp, 
            "nOperations": 18,
            "nDestinations": nRegisters,
            "inputSize": inputSize, 
            "initMaxProgSize": initMaxProgSize,
            "rampantGen": rampancy[0], 
            "rampantMin": rampancy[1], 
            "rampantMax": rampancy[2], 
            "idCountTeam": 0, 
            "idCountLearner": 0, 
            "idCountProgram": 0
        }
        self.memVars:       dict = {
            "frameNum": 0,
            "task": 'task',
            "memMatrix": self.memMatrix,
        }
        self.nOperations = 18
        # self.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        if state: self.setMemories(state)

    def _initialize(self, _state=None):
        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            # m1,m2 = random.choices(range(len(self.memoryCodes)), 2)

            l1 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                memoryObj=self.__class__.MemoryObject(state=_state),
                numRegisters=self.nRegisters)
            
            l2 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                memoryObj=self.__class__.MemoryObject(state=_state),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = self.__class__.Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):

                learner = self.__class__.Learner(
                    initParams=self.mutateParams,
                    program=self.__class__.Program(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    memoryObj=self.__class__.MemoryObject(state=_state),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _select(self, extraTeams=None, task='task'):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt[task])
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
            # else: # delete MemoryObject
            #     self.__class__.MemoryObject.memories.referenced[cursor.memoryObj.memoryCode]-=1

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0] 

    def _generate(self, extraTeams=None, _states=None, _rewards=None, _unexpectancies=None):
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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getMemoryTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            if not _states is None and not _rewards is None and not _unexpectancies is None: 
                states = np.array(_states)
                unexpectancies = np.array(_unexpectancies)
                unexpectancies = sigmoid(unexpectancies)*1000
                # randome choice story by rewards
                state = random.choices(range(len(states)), unexpectancies)[0]

                child.addLearner(self.__class__.Learner(memoryObj=self.__class__.MemoryObject(state=states[state], reward=_rewards[state])))
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
        self.__class__.MemoryObject.memories.oblivion(memory_code_list)

        self.generation += 1

    def getAgents(self, sortTasks=[], multiTaskType='min', skipTasks=[], task='task'):
        self.memVars['task']=task
        # remove those that get skipped
        rTeams = [rt for rt in self.rootTeams
                if len(skipTasks) == 0
                        or any(task not in rt.outcomes for task in skipTasks)]

        if len(sortTasks) == 0: # just get all
            return [self.__class__.Agent(team, num=i, memVars=self.memVars)
                    for i,team in enumerate(rTeams)]
        else:

            if len(sortTasks) == 1:
                rTeams = [t for t in rTeams if sortTasks[0] in t.outcomes]
                # return teams sorted by the outcome
                return [self.__class__.Agent(team, num=i, memVars=self.memVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.outcomes[sortTasks[0]], reverse=True))]

            else:
                # apply scores/fitness to root teams
                self._scoreIndividuals(sortTasks, multiTaskType=multiTaskType, doElites=False)
                # return teams sorted by fitness
                return [self.__class__.Agent(team, num=i, memVars=self.memVars)
                        for i,team in enumerate(sorted(rTeams,
                                        key=lambda tm: tm.fitness, reverse=True))]

    def getEliteAgent(self, task):
        
        teams = [t for t in self.teams if task in t.outcomes]

        return self.__class__.Agent(max([tm for tm in teams],
                        key=lambda t: t.outcomes[task]),
                        num=0, memVars=self.memVars)

    def setMemories(self, state):
        assert isinstance(state, np.ndarray)

        for _ in range(self.initMaxTeamSize):
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            self.__class__.MemoryObject.memories.append(key, state)
        self._initialize(_state=state)
        return self.__class__.MemoryObject.memories

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None, _states=None, _rewards=None, _unexpectancies=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams, task=tasks[0]) # select individuals to keep
        self._generate(extraTeams, _states=_states, _rewards=_rewards, _unexpectancies=_unexpectancies) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def save(self, fileName):
        self._memories = self.__class__.MemoryObject.memories
        pickle.dump(self, open(f'log/{fileName}.pickle', 'wb'))

    @classmethod
    def load(cls, fileName:str):
        trainer = pickle.load(open(f'log/{fileName}.pickle', 'rb'))
        assert isinstance(trainer, cls), f'this file is not {cls}'

        trainer.MemoryObject.memories = trainer._memories
        return trainer

class Trainer2_1(Trainer2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent2
            from _tpg.team import Team2_1
            from _tpg.learner import Learner2_1
            from _tpg.program import Program2
            from _tpg.memory_object import MemoryObject

            cls._instance = True
            cls.Agent = Agent2
            cls.Team = Team2_1
            cls.Learner = Learner2_1
            cls.Program = Program2
            cls.MemoryObject = MemoryObject
            cls.MemoryObject._nan = cls.MemoryObject()

        return super().__new__(cls, *args, **kwargs)

    def _select(self, extraTeams=None, task='task'):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt[task])
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

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0] 

    def _generate(self, extraTeams=None, _states=None, _rewards=None, _unexpectancies=None):
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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getMemoryTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            if not _states is None and not _rewards is None and not _unexpectancies is None: 
                states = np.array(_states)
                unexpectancies = np.array(_unexpectancies)
                unexpectancies = sigmoid(unexpectancies)*1000
                # randome choice story by rewards
                state = random.choices(range(len(states)), unexpectancies)[0]

                child.addLearner(self.__class__.Learner(memoryObj=self.__class__.MemoryObject(state=states[state], reward=_rewards[state])))
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
        self.__class__.MemoryObject.memories.oblivion(memory_code_list)

        self.generation += 1

class Trainer2_2(Trainer2_1):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent2_1
            from _tpg.team import Team2_2
            from _tpg.learner import Learner2_2
            from _tpg.program import Program2_1
            from _tpg.memory_object import MemoryObject2

            cls._instance = True
            cls.Agent = Agent2_1
            cls.Team = Team2_2
            cls.Learner = Learner2_2
            cls.Program = Program2_1
            cls.MemoryObject = MemoryObject2
            cls.MemoryObject._nan = cls.MemoryObject()

        return super().__new__(cls, *args, **kwargs)

    def _select(self, extraTeams=None, task='task'):
        # print('em:',[rt.fitness for rt in self.rootTeams])
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

        # Finaly, purge the orphans
        # AtomicActionのLearnerはどのように生成すれば良いのだろうか？ -> actionObj.mutate()による
        self.learners = [learner for learner in self.learners if learner.numTeamsReferencing() > 0] 

    def _generate(self, extraTeams=None, _states=None, _rewards=None, _unexpectancies=None):
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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getMemoryTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

            if not _states is None and not _rewards is None and not _unexpectancies is None: 
                states = np.array(_states)
                unexpectancies = np.array(_unexpectancies)
                unexpectancies = sigmoid(unexpectancies)*1000
                # randome choice story by rewards
                state = random.choices(range(len(states)), unexpectancies)[0]

                child.addLearner(self.__class__.Learner(memoryObj=self.__class__.MemoryObject(state=states[state], reward=_rewards[state])))
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
        self.__class__.MemoryObject.memories.oblivion(memory_code_list)

        self.generation += 1

    def setMemories(self, state):
        assert isinstance(state, np.ndarray)

        for _ in range(self.initMaxTeamSize):
            key = np.random.choice(range(state.size), random.randint(1, state.size-1))
            self.__class__.MemoryObject.memories.append(_state=state, _key=key)
        self._initialize(_state=state)
        return self.__class__.MemoryObject.memories

class Trainer2_3(Trainer2_2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent2_3
            from _tpg.team import Team2_3
            from _tpg.learner import Learner2_3
            from _tpg.program import Program2_3
            # from _tpg.memory_object import Qualia
            from _tpg.memory_object import Hippocampus

            cls._instance = True
            cls.Agent = Agent2_3
            cls.Team = Team2_3
            cls.Learner = Learner2_3
            cls.Program = Program2_3
            cls.MemoryObject = Hippocampus
            cls.MemoryObject()

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
        primitive=None, 
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
        # memType="def", 
        memMatrixShape:tuple=(100,8),       # *
        rampancy:tuple=(0,0,0),
        # operationSet:str="custom", 
        # traversal:str="team", 
        prevPops=None, mutatePrevs=True,
        initMaxActProgSize:int=6,           # *
        nMemRegisters:int=4
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
        # if traversal not in ["team", "learner"]:    raise Exception("Invalid traversal")
        
        '''
        Gap must be a float greater than 0 but less than or equal to 1
        '''
        if type(gap) is not float or gap == 0.0 or gap > 1.0:   raise Exception("gap must be a float greater than 0 but less than or equal to 1")


        # Validate Operation Set
        # if operationSet not in ["def", "full", "robo", "custom"]:   raise Exception("Invalid operation set")

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

        # first store actions properly
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


        # whether to keep elites
        self.doElites = doElites

        # if memType == "None":   self.memType = None
        # self.memType = memType
        # self.memMatrixShape = memMatrixShape
        self.memMatrix = np.zeros(shape=memMatrixShape)

        self.rampancy = rampancy

        # self.operationSet = operationSet

        # self.traversal = traversal

        self.initMaxActProgSize = initMaxActProgSize
        # ensure nActRegisters is larger than the largest action length
        # if self.doReal: nActRegisters = max(max(self.actionLengths), nActRegisters)
        self.nMemRegisters = nMemRegisters

        # core components of TPG
        self.teams:     list = []
        self.rootTeams: list = []
        self.learners:  list = []
        self.elites:    list = [] # save best at each task

        self.generation = 0 # track this

        # these are to be filled in by the configurer after
        self.mutateParams:  dict = {
            "generation": self.generation,
            "maxTeamSize": maxTeamSize,
            "pLrnAdd": pLrnAdd, 
            "pLrnDel": pLrnDel, 
            "pLrnMut": pLrnMut,
            "pProgMut": pProgMut, 
            "pMemAtom": pMemAtom, 
            "pMemMut": pMemMut, 
            "pInstAdd": pInstAdd, 
            "pInstDel": pInstDel, 
            "pInstMut": pInstMut,
            "pInstSwp": pInstSwp, 
            "pExtinction": 0.8, 
            "nOperations": 18,
            "nDestinations": nRegisters,
            "inputSize": inputSize, 
            "initMaxProgSize": initMaxProgSize,
            "rampantGen": rampancy[0], 
            "rampantMin": rampancy[1], 
            "rampantMax": rampancy[2], 
            "idCountTeam": 0, 
            "idCountLearner": 0, 
            "idCountProgram": 0
        }
        self.memVars:       dict = {
            "frameNum": 0,
            "task": 'task',
            "memMatrix": self.memMatrix,
        }
        self.nOperations = 18
        # self.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        if primitive: self.setMemories(primitive)

    def _initialize(self):
        for _ in range(self.teamPopSize):
            # create 2 unique actions and learners
            # m1,m2 = random.choices(range(len(self.memoryCodes)), 2)

            l1 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)
            
            l2 = self.__class__.Learner(
                initParams=self.mutateParams,
                program=self.__class__.Program(
                    maxProgramLength=self.initMaxProgSize,
                    nOperations=self.nOperations,
                    nDestinations=self.nRegisters,
                    inputSize=self.inputSize,
                    initParams=self.mutateParams),
                numRegisters=self.nRegisters)

            # save learner population
            self.learners.append(l1)
            self.learners.append(l2)

            # create team and add initial learners
            team = self.__class__.Team(initParams=self.mutateParams)
            team.addLearner(l1)
            team.addLearner(l2)

            # add more learners
            moreLearners = random.randint(0, self.initMaxTeamSize-2)
            for __ in range(moreLearners):

                learner = self.__class__.Learner(
                    initParams=self.mutateParams,
                    program=self.__class__.Program(
                        maxProgramLength=self.initMaxProgSize,
                        nOperations=self.nOperations,
                        nDestinations=self.nRegisters,
                        inputSize=self.inputSize,
                        initParams=self.mutateParams),
                    memoryObj=self.__class__.MemoryObject(),  
                    numRegisters=self.nRegisters)

                team.addLearner(learner)
                self.learners.append(learner)

            # save to team populations
            self.teams.append(team)
            self.rootTeams.append(team)

    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness*rt.extinction)
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
                cursor.memoryObj.teamMemory.inLearners.remove(cursor.id)

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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # childs.sequence to memoryObj
            child.addSequence()

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getMemoryTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

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
            team.extinction*=1.01

            # maybe make root team
            if team.numLearnersReferencing() == 0 or team in self.elites:
                self.rootTeams.append(team)

        for lrnr in self.learners:
            if lrnr.isMemoryAtomic():
                memory_code_list.add(lrnr.memoryObj.memoryCode)

        memory_code_list = list(memory_code_list)
        self.__class__.MemoryObject.memories.oblivion(memory_code_list)

        self.generation += 1

    def _scoreIndividuals(self, tasks, multiTaskType='min'):
        # handle generation of new elites, typically just done in evolution
        assert isinstance(tasks, list), f'{tasks} is not list'

        if self.doElites:
            # get the best agent at each task
            self.elites = [] # clear old elites
            for task in tasks:
                self.elites.append(max([rt for rt in self.rootTeams],
                                        key=lambda t: t[task]))

        if len(tasks) == 1: # single fitness
            for rt in self.rootTeams:
                rt.fitness = rt[tasks[0]]
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
            mins.append(min([team[task] for team in self.rootTeams]))
            maxs.append(max([team[task] for team in self.rootTeams]))

        # assign fitness
        if multiTaskType == 'min':
            for rt in self.rootTeams:
                rt.fitness = min([(rt[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'max':
            for rt in self.rootTeams:
                rt.fitness = max([(rt[task]-mins[i])/(maxs[i]-mins[i])
                        for i,task in enumerate(tasks)])
        elif multiTaskType == 'average':
            for rt in self.rootTeams:
                scores = [(rt[task]-mins[i])/(maxs[i]-mins[i])
                            for i,task in enumerate(tasks)]
                rt.fitness = sum(scores)/len(scores)

    def evolve(self, tasks=['task'], multiTaskType='min', extraTeams=None):
        self._scoreIndividuals(
            tasks, 
            multiTaskType=multiTaskType,
        ) # assign scores to individuals
        self._saveFitnessStats() # save fitness stats
        self._select(extraTeams) # select individuals to keep
        self._generate(extraTeams) # create new individuals from those kept
        self._nextEpoch() # set up for next generation

    def setMemories(self, _state):
        assert isinstance(_state, np.ndarray)

        for _ in range(self.initMaxTeamSize):
            state = np.array([np.nan]*_state.size)
            key = np.random.choice(range(_state.size), random.randint(1, _state.size-1))
            state[key] = _state[key]
            self.__class__.MemoryObject.memories.append(_sequence=state)
        self._initialize()
        return self.__class__.MemoryObject.memories
    
    # @classmethod
    @property
    def elite(self):
        teams = [t for t in self.teams if self.actVars['task'] in t['task']]

        return self.__class__.Agent(max([tm for tm in teams],
                        key=lambda t: t.outcomes[self.actVars['task']]),
                        num=0, actVars=self.actVars)

class Trainer2_4(Trainer2_3):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.agent import Agent2_3
            from _tpg.team import Team2_3
            from _tpg.learner import Learner2_3
            from _tpg.program import Program2_3
            from _tpg.memory_object import Hippocampus

            cls._instance = True
            cls.Agent = Agent2_3
            cls.Team = Team2_3
            cls.Learner = Learner2_3
            cls.Program = Program2_3
            cls.MemoryObject = Hippocampus
            cls.MemoryObject()

        return super().__new__(cls, *args, **kwargs)

    def _select(self, extraTeams=None):

        rankedTeams = sorted(self.rootTeams, key=lambda rt: rt.fitness*rt.extinction)
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
                cursor.memoryObj.teamMemory.inLearners.remove(cursor.id)

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
            child = parent.clone

            # child starts just like parent
            # for learner in parent.learners: child.addLearner(learner)

            # then mutates
            # child.mutate(self.mutateParams, oLearners, oTeams)
            _, __, new_learners = child.mutate(self.mutateParams, oLearners, oTeams)

            # childs.sequence to memoryObj
            child.addSequence()

            # then clone the referenced rootTeams
            for new_learner in new_learners:
                tm = new_learner.getMemoryTeam()
                if tm in self.rootTeams:
                    clone = tm.clone
                    self.teams.append(clone)
                    
                    assert not clone in self.rootTeams and tm in self.rootTeams, 'prease clone remove from rootTeams'

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
            team.extinction*=1.01

            # maybe make root team
            if team.extinction > self.mutateParams['pExtinction'] or team in self.elites:
                self.rootTeams.append(team)

        for lrnr in self.learners:
            if lrnr.isMemoryAtomic():
                memory_code_list.add(lrnr.memoryObj.memoryCode)

        memory_code_list = list(memory_code_list)
        self.__class__.MemoryObject.memories.oblivion(memory_code_list)

        self.generation += 1
