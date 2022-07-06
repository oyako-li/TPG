from _tpg.configuration.conf_agent import ConfAgent, ConfAgent1, ConfAgent11, ConfAgent2
from _tpg.configuration.conf_team import ConfTeam, ConfTeam1, ConfTeam11, ConfTeam2
from _tpg.configuration.conf_learner import ConfLearner, ConfLearner1, ConfLearner11, ConfLearner2
from _tpg.configuration.conf_action_object import ConfActionObject, ConfActionObject11, ConfActionObject1
from _tpg.configuration.conf_program import ConfProgram, ConfProgram1, ConfProgram11, ConfProgram2
# from _tpg.configuration.conf_emulator import ConfEmulator

import numpy as np

"""
Contains the ability to swap out different functions and add arbitrary variables
to different classes willy nilly to support whatever functionality is desired.
Such as Memory or real actions. Default configuration is no memory and discrete
actions.
"""
def configure(trainer, Trainer, Agent, Team, Learner, ActionObject, Program,
        doMemory, memType, doReal, operationSet, traversal):

    # keys and values used in key value pairs for suplementary function args
    # for mutation and creation
    mutateParamKeys = ["generation", "maxTeamSize", "pLrnDel", "pLrnAdd", "pLrnMut",
        "pProgMut", "pActMut", "pActAtom", "pInstDel", "pInstAdd", "pInstSwp", "pInstMut",
        "actionCodes", "nDestinations", "inputSize", "initMaxProgSize",
        "rampantGen", "rampantMin", "rampantMax", "idCountTeam", "idCountLearner", "idCountProgram"]
    mutateParamVals = [trainer.generation, trainer.maxTeamSize, trainer.pLrnDel, trainer.pLrnAdd, trainer.pLrnMut,
        trainer.pProgMut, trainer.pActMut, trainer.pActAtom, trainer.pInstDel, trainer.pInstAdd, trainer.pInstSwp, trainer.pInstMut,
        trainer.actionCodes, trainer.nRegisters, trainer.inputSize, trainer.initMaxProgSize,
        trainer.rampancy[0], trainer.rampancy[1], trainer.rampancy[2], 0, 0, 0]

    # additional stuff for act, like memory matrix possible
    actVarKeys = ["frameNum"]
    actVarVals = [0]

    # before doing any special configuration, set all methods to defaults
    _configureDefaults(trainer, Trainer, Agent, Team, Learner, ActionObject, Program)

    # configure Program execution stuff, affected by memory and operations set
    _configureProgram(trainer, Learner, Program, actVarKeys, actVarVals, mutateParamKeys, mutateParamVals, doMemory, memType, operationSet)

    # configure stuff for using real valued actions
    if doReal: _configureRealAction(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory)

    # do learner traversal
    if traversal == "learner": _configureLearnerTraversal(trainer, Agent, Team, actVarKeys, actVarVals)

    trainer.mutateParams = dict(zip(mutateParamKeys, mutateParamVals))
    trainer.actVars = dict(zip(actVarKeys, actVarVals))

"""
For each class in TPG, sets the functions to their defaults.
"""
def _configureDefaults(trainer, Trainer, Agent, Team, Learner, ActionObject, Program):
    # set trainer functions
    # TODO: add learner configurable

    # set agent functions
    Agent.__init__ = ConfAgent.init_def
    Agent.act = ConfAgent.act_def
    Agent.reward = ConfAgent.reward_def
    Agent.taskDone = ConfAgent.taskDone_def
    Agent.saveToFile = ConfAgent.saveToFile_def

    # set team functions
    Team.__init__ = ConfTeam.init_def
    Team.act = ConfTeam.act_def
    Team.addLearner = ConfTeam.addLearner_def
    Team.removeLearner = ConfTeam.removeLearner_def
    Team.removeLearners = ConfTeam.removeLearners_def
    Team.numAtomicActions = ConfTeam.numAtomicActions_def
    Team.mutate = ConfTeam.mutate_def

    # set learner functions
    Learner.__init__ = ConfLearner.init_def
    Learner.bid = ConfLearner.bid_def
    Learner.getAction = ConfLearner.getAction_def
    Learner.getActionTeam = ConfLearner.getActionTeam_def
    Learner.isActionAtomic = ConfLearner.isActionAtomic_def
    Learner.mutate = ConfLearner.mutate_def
    Learner.clone = ConfLearner.clone_def

    # set action object functions
    ActionObject.__init__ = ConfActionObject.init_def
    ActionObject.getAction = ConfActionObject.getAction_def
    ActionObject.isAtomic = ConfActionObject.isAtomic_def
    ActionObject.mutate = ConfActionObject.mutate_def
    ActionObject._actions = trainer.actionCodes


    # set program functions
    Program.__init__ = ConfProgram.init_def
    Program.execute = ConfProgram.execute_def
    Program.mutate = ConfProgram.mutate_def
    Program.memWriteProbFunc = ConfProgram.memWriteProb_def

    # let trainer know what functions are set for each one
    
    trainer.functionsDict["Agent"] = {
        "init": "def",
        "act": "def",
        "reward": "def",
        "taskDone": "def",
        "saveToFile": "def"
    }
    trainer.functionsDict["Team"] = {
        "init": "def",
        "act": "def",
        "addLearner": "def",
        "removeLearner": "def",
        "removeLearners": "def",
        "numAtomicActions": "def",
        "mutate": "def"
    }
    trainer.functionsDict["Learner"] = {
        "init": "def",
        "bid": "def",
        "getAction": "def",
        "getActionTeam": "def",
        "isActionAtomic": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["ActionObject"] = {
        "init": "def",
        "getAction": "def",
        "getRealAction": "None",
        "isAtomic": "def",
        "mutate": "def"
    }
    trainer.functionsDict["Program"] = {
        "init": "def",
        "execute": "def",
        "mutate": "def",
        "memWriteProbFunc": "def"
    }

"""
Decides the operations and functions to be used in program execution.
"""
def _configureProgram(trainer, Learner, Program, actVarKeys, actVarVals,
        mutateParamKeys, mutateParamVals, doMemory, memType, operationSet):
    # change functions as needed
    if doMemory:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram.execute_mem
            trainer.functionsDict["Program"]["execute"] = "mem"
            trainer.nOperations = 7
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "full":
            Program.execute = ConfProgram.execute_mem_full
            trainer.functionsDict["Program"]["execute"] = "mem_full"
            trainer.nOperations = 10
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "robo":
            Program.execute = ConfProgram.execute_mem_robo
            trainer.functionsDict["Program"]["execute"] = "mem_robo"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "custom":
            Program.execute = ConfProgram.execute_mem_custom
            trainer.functionsDict["Program"]["execute"] = "mem_custom"
            trainer.nOperations = 18
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        # select appropriate memory write function
        if memType == "cauchy1":
            Program.memWriteProbFunc = ConfProgram.memWriteProb_cauchy1
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchy1"
        elif memType == "cauchyHalf":
            Program.memWriteProbFunc = ConfProgram.memWriteProb_cauchyHalf
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchyHalf"
        else:
            Program.memWriteProbFunc = ConfProgram.memWriteProb_def
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "def"

        # change bid function to accomodate additional parameters needed for memory
        Learner.bid = ConfLearner.bid_mem
        trainer.functionsDict["Learner"]["bid"] = "mem"

        # trainer needs to have memory
        trainer.memMatrix = np.zeros(shape=trainer.memMatrixShape)
        # agents need access to memory too, and to pass through act
        actVarKeys += ["memMatrix"]
        actVarVals += [trainer.memMatrix]

    else:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram.execute_def
            trainer.functionsDict["Program"]["execute"] = "def"
            trainer.nOperations = 5
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG"]
        elif operationSet == "full":
            Program.execute = ConfProgram.execute_full
            trainer.functionsDict["Program"]["execute"] = "full"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP"]
        elif operationSet == "robo":
            Program.execute = ConfProgram.execute_robo
            trainer.functionsDict["Program"]["execute"] = "robo"
            trainer.nOperations = 6
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS"]
        elif operationSet == "custom":
            Program.execute = ConfProgram.execute_custom
            trainer.functionsDict["Program"]["execute"] = "custom"
            trainer.nOperations = 16
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS"]


        Learner.bid = ConfLearner.bid_def
        trainer.functionsDict["Learner"]["bid"] = "def"

    mutateParamKeys += ["nOperations"]
    mutateParamVals += [trainer.nOperations]

"""
Make the appropriate changes needed to be able to use real actions.
"""
def _configureRealAction(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory):
    # change functions as needed
    ActionObject.__init__ = ConfActionObject.init_real
    trainer.functionsDict["ActionObject"]["init"] = "real"
    ActionObject.getAction = ConfActionObject.getAction_real
    trainer.functionsDict["ActionObject"]["getAction"] = "real"
    if doMemory:
        ActionObject.getRealAction = ConfActionObject.getRealAction_real_mem
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real_mem"
    else:
        ActionObject.getRealAction = ConfActionObject.getRealAction_real
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real"
    ActionObject.mutate = ConfActionObject.mutate_real
    trainer.functionsDict["ActionObject"]["mutate"] = "real"

    # mutateParams needs to have lengths of actions and act program
    mutateParamKeys += ["actionLengths", "initMaxActProgSize", "nActRegisters"]
    mutateParamVals += [trainer.actionLengths, trainer.initMaxActProgSize, trainer.nActRegisters]

"""
Switch to learner traversal.
"""
def _configureLearnerTraversal(trainer, Agent, Team, actVarKeys, actVarVals):
    Team.act = ConfTeam.act_learnerTrav
    trainer.functionsDict["Team"]["act"] = "learnerTrav"


def configure1(trainer, Trainer, Agent, Team, Learner, ActionObject, Program,
        doMemory, memType, doReal, operationSet, traversal):

    # keys and values used in key value pairs for suplementary function args
    # for mutation and creation
    mutateParamKeys = ["generation", "maxTeamSize", "pLrnDel", "pLrnAdd", "pLrnMut",
        "pProgMut", "pActMut", "pActAtom", "pInstDel", "pInstAdd", "pInstSwp", "pInstMut",
        "actionCodes", "nDestinations", "inputSize", "initMaxProgSize",
        "rampantGen", "rampantMin", "rampantMax", "idCountTeam", "idCountLearner", "idCountProgram"]
    mutateParamVals = [trainer.generation, trainer.maxTeamSize, trainer.pLrnDel, trainer.pLrnAdd, trainer.pLrnMut,
        trainer.pProgMut, trainer.pActMut, trainer.pActAtom, trainer.pInstDel, trainer.pInstAdd, trainer.pInstSwp, trainer.pInstMut,
        trainer.actionCodes, trainer.nRegisters, trainer.inputSize, trainer.initMaxProgSize,
        trainer.rampancy[0], trainer.rampancy[1], trainer.rampancy[2], 0, 0, 0]

    # additional stuff for act, like memory matrix possible
    actVarKeys = ["frameNum"]
    actVarVals = [0]

    # before doing any special configuration, set all methods to defaults
    _configureDefaults1(trainer, Trainer, Agent, Team, Learner, ActionObject, Program)

    # configure Program execution stuff, affected by memory and operations set
    _configureProgram1(trainer, Learner, Program, actVarKeys, actVarVals, mutateParamKeys, mutateParamVals, doMemory, memType, operationSet)

    # configure stuff for using real valued actions
    if doReal:  _configureRealAction1(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory)

    # do learner traversal
    if traversal == "learner": _configureLearnerTraversal1(trainer, Agent, Team, actVarKeys, actVarVals)

    trainer.mutateParams = dict(zip(mutateParamKeys, mutateParamVals))
    trainer.actVars = dict(zip(actVarKeys, actVarVals))

"""
For each class in TPG, sets the functions to their defaults.
"""
def _configureDefaults1(trainer, Trainer, Agent, Team, Learner, ActionObject, Program):
    # set trainer functions
    # TODO: add learner configurable

    # set agent functions
    Agent.__init__ = ConfAgent1.init_def
    Agent.act = ConfAgent1.act_def
    Agent.reward = ConfAgent1.reward_def
    Agent.taskDone = ConfAgent1.taskDone_def
    Agent.saveToFile = ConfAgent1.saveToFile_def

    # set team functions
    Team.__init__ = ConfTeam1.init_def
    Team.act = ConfTeam1.act_def
    Team.addLearner = ConfTeam1.addLearner_def
    Team.removeLearner = ConfTeam1.removeLearner_def
    Team.removeLearners = ConfTeam1.removeLearners_def
    Team.numAtomicActions = ConfTeam1.numAtomicActions_def
    Team.mutate = ConfTeam1.mutate_def
    Team.clone = ConfTeam1.clone_def

    # set learner functions
    Learner.__init__ = ConfLearner1.init_def
    Learner.bid = ConfLearner1.bid_def
    Learner.getAction = ConfLearner1.getAction_def
    Learner.getActionTeam = ConfLearner1.getActionTeam_def
    Learner.isActionAtomic = ConfLearner1.isActionAtomic_def
    Learner.mutate = ConfLearner1.mutate_def
    Learner.clone = ConfLearner1.clone_def


    # set action object functions
    ActionObject.__init__ = ConfActionObject1.init_def
    ActionObject.getAction = ConfActionObject1.getAction_def
    ActionObject.isAtomic = ConfActionObject1.isAtomic_def
    ActionObject.mutate = ConfActionObject1.mutate_def
    ActionObject._actions = trainer.actionCodes


    # set program functions
    Program.__init__ = ConfProgram1.init_def
    Program.execute = ConfProgram1.execute_def
    Program.mutate = ConfProgram1.mutate_def
    Program.memWriteProbFunc = ConfProgram1.memWriteProb_def

    # let trainer know what functions are set for each one
    
    trainer.functionsDict["Agent"] = {
        "init": "def",
        "act": "def",
        "reward": "def",
        "taskDone": "def",
        "saveToFile": "def"
    }
    trainer.functionsDict["Team"] = {
        "init": "def",
        "act": "def",
        "addLearner": "def",
        "removeLearner": "def",
        "removeLearners": "def",
        "numAtomicActions": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["Learner"] = {
        "init": "def",
        "bid": "def",
        "getAction": "def",
        "getActionTeam": "def",
        "isActionAtomic": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["ActionObject"] = {
        "init": "def",
        "getAction": "def",
        "getRealAction": "None",
        "isAtomic": "def",
        "mutate": "def"
    }
    trainer.functionsDict["Program"] = {
        "init": "def",
        "execute": "def",
        "mutate": "def",
        "memWriteProbFunc": "def"
    }

"""
Decides the operations and functions to be used in program execution.
"""
def _configureProgram1(trainer, Learner, Program, actVarKeys, actVarVals,
        mutateParamKeys, mutateParamVals, doMemory, memType, operationSet):
    # change functions as needed
    if doMemory:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram1.execute_mem
            trainer.functionsDict["Program"]["execute"] = "mem"
            trainer.nOperations = 7
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "full":
            Program.execute = ConfProgram1.execute_mem_full
            trainer.functionsDict["Program"]["execute"] = "mem_full"
            trainer.nOperations = 10
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "robo":
            Program.execute = ConfProgram1.execute_mem_robo
            trainer.functionsDict["Program"]["execute"] = "mem_robo"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "custom":
            Program.execute = ConfProgram1.execute_mem_custom
            trainer.functionsDict["Program"]["execute"] = "mem_custom"
            trainer.nOperations = 18
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        # select appropriate memory write function
        if memType == "cauchy1":
            Program.memWriteProbFunc = ConfProgram1.memWriteProb_cauchy1
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchy1"
        elif memType == "cauchyHalf":
            Program.memWriteProbFunc = ConfProgram1.memWriteProb_cauchyHalf
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchyHalf"
        else:
            Program.memWriteProbFunc = ConfProgram1.memWriteProb_def
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "def"

        # change bid function to accomodate additional parameters needed for memory
        Learner.bid = ConfLearner1.bid_mem
        trainer.functionsDict["Learner"]["bid"] = "mem"

        # trainer needs to have memory
        trainer.memMatrix = np.zeros(shape=trainer.memMatrixShape)
        # agents need access to memory too, and to pass through act
        actVarKeys += ["memMatrix"]
        actVarVals += [trainer.memMatrix]

    else:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram1.execute_def
            trainer.functionsDict["Program"]["execute"] = "def"
            trainer.nOperations = 5
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG"]
        elif operationSet == "full":
            Program.execute = ConfProgram1.execute_full
            trainer.functionsDict["Program"]["execute"] = "full"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP"]
        elif operationSet == "robo":
            Program.execute = ConfProgram1.execute_robo
            trainer.functionsDict["Program"]["execute"] = "robo"
            trainer.nOperations = 6
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS"]
        elif operationSet == "custom":
            Program.execute = ConfProgram1.execute_custom
            trainer.functionsDict["Program"]["execute"] = "custom"
            trainer.nOperations = 16
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS"]


        Learner.bid = ConfLearner1.bid_def
        trainer.functionsDict["Learner"]["bid"] = "def"

    mutateParamKeys += ["nOperations"]
    mutateParamVals += [trainer.nOperations]

"""
Make the appropriate changes needed to be able to use real actions.
"""
def _configureRealAction1(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory):
    # change functions as needed
    ActionObject.__init__ = ConfActionObject1.init_real
    trainer.functionsDict["ActionObject"]["init"] = "real"
    ActionObject.getAction = ConfActionObject1.getAction_real
    trainer.functionsDict["ActionObject"]["getAction"] = "real"
    if doMemory:
        ActionObject.getRealAction = ConfActionObject1.getRealAction_real_mem
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real_mem"
    else:
        ActionObject.getRealAction = ConfActionObject1.getRealAction_real
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real"
    ActionObject.mutate = ConfActionObject1.mutate_real
    trainer.functionsDict["ActionObject"]["mutate"] = "real"

    # mutateParams needs to have lengths of actions and act program
    mutateParamKeys += ["actionLengths", "initMaxActProgSize", "nActRegisters"]
    mutateParamVals += [trainer.actionLengths, trainer.initMaxActProgSize, trainer.nActRegisters]

"""
Switch to learner traversal.
"""
def _configureLearnerTraversal1(trainer, Agent, Team, actVarKeys, actVarVals):
    Team.act = ConfTeam1.act_learnerTrav
    trainer.functionsDict["Team"]["act"] = "learnerTrav"

def configure11(trainer, Trainer, Agent, Team, Learner, ActionObject, Program,
        doMemory, memType, doReal, operationSet, traversal):

    # keys and values used in key value pairs for suplementary function args
    # for mutation and creation
    mutateParamKeys = ["generation", "maxTeamSize", "pLrnDel", "pLrnAdd", "pLrnMut",
        "pProgMut", "pActMut", "pActAtom", "pInstDel", "pInstAdd", "pInstSwp", "pInstMut",
        "actionCodes", "nDestinations", "inputSize", "initMaxProgSize",
        "rampantGen", "rampantMin", "rampantMax", "idCountTeam", "idCountLearner", "idCountProgram"]
    mutateParamVals = [trainer.generation, trainer.maxTeamSize, trainer.pLrnDel, trainer.pLrnAdd, trainer.pLrnMut,
        trainer.pProgMut, trainer.pActMut, trainer.pActAtom, trainer.pInstDel, trainer.pInstAdd, trainer.pInstSwp, trainer.pInstMut,
        trainer.actionCodes, trainer.nRegisters, trainer.inputSize, trainer.initMaxProgSize,
        trainer.rampancy[0], trainer.rampancy[1], trainer.rampancy[2], 0, 0, 0]

    # additional stuff for act, like memory matrix possible
    actVarKeys = ["frameNum"]
    actVarVals = [0]

    # before doing any special configuration, set all methods to defaults
    _configureDefaults11(trainer, Trainer, Agent, Team, Learner, ActionObject, Program)

    # configure Program execution stuff, affected by memory and operations set
    _configureProgram11(trainer, Learner, Program, actVarKeys, actVarVals, mutateParamKeys, mutateParamVals, doMemory, memType, operationSet)

    # configure stuff for using real valued actions
    if doReal:  _configureRealAction11(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory)

    # do learner traversal
    if traversal == "learner": _configureLearnerTraversal11(trainer, Agent, Team, actVarKeys, actVarVals)

    trainer.mutateParams = dict(zip(mutateParamKeys, mutateParamVals))
    trainer.actVars = dict(zip(actVarKeys, actVarVals))

"""
For each class in TPG, sets the functions to their defaults.
"""
def _configureDefaults11(trainer, Trainer, Agent, Team, Learner, ActionObject, Program):
    # set trainer functions
    # TODO: add learner configurable

    # set agent functions
    Agent.__init__ = ConfAgent11.init_def
    Agent.act = ConfAgent11.act_def
    Agent.reward = ConfAgent11.reward_def
    Agent.taskDone = ConfAgent11.taskDone_def
    Agent.saveToFile = ConfAgent11.saveToFile_def

    # set team functions
    Team.__init__ = ConfTeam11.init_def
    Team.act = ConfTeam11.act_def
    Team.addLearner = ConfTeam11.addLearner_def
    Team.removeLearner = ConfTeam11.removeLearner_def
    Team.removeLearners = ConfTeam11.removeLearners_def
    Team.numAtomicActions = ConfTeam11.numAtomicActions_def
    Team.mutate = ConfTeam11.mutate_def
    Team.clone = ConfTeam11.clone_def

    # set learner functions
    Learner.__init__ = ConfLearner11.init_def
    Learner.bid = ConfLearner11.bid_def
    Learner.getAction = ConfLearner11.getAction_def
    Learner.getActionTeam = ConfLearner11.getActionTeam_def
    Learner.isActionAtomic = ConfLearner11.isActionAtomic_def
    Learner.mutate = ConfLearner11.mutate_def
    Learner.clone = ConfLearner11.clone_def


    # set action object functions
    ActionObject.__init__ = ConfActionObject11.init_def
    ActionObject.getAction = ConfActionObject11.getAction_def
    ActionObject.isAtomic = ConfActionObject11.isAtomic_def
    ActionObject.mutate = ConfActionObject11.mutate_def
    ActionObject.actions = trainer.actionCodes


    # set program functions
    Program.__init__ = ConfProgram11.init_def
    Program.execute = ConfProgram11.execute_def
    Program.mutate = ConfProgram11.mutate_def
    Program.memWriteProbFunc = ConfProgram11.memWriteProb_def

    # let trainer know what functions are set for each one
    
    trainer.functionsDict["Agent"] = {
        "init": "def",
        "act": "def",
        "reward": "def",
        "taskDone": "def",
        "saveToFile": "def"
    }
    trainer.functionsDict["Team"] = {
        "init": "def",
        "act": "def",
        "addLearner": "def",
        "removeLearner": "def",
        "removeLearners": "def",
        "numAtomicActions": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["Learner"] = {
        "init": "def",
        "bid": "def",
        "getAction": "def",
        "getActionTeam": "def",
        "isActionAtomic": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["ActionObject"] = {
        "init": "def",
        "getAction": "def",
        "getRealAction": "None",
        "isAtomic": "def",
        "mutate": "def"
    }
    trainer.functionsDict["Program"] = {
        "init": "def",
        "execute": "def",
        "mutate": "def",
        "memWriteProbFunc": "def"
    }

"""
Decides the operations and functions to be used in program execution.
"""
def _configureProgram11(trainer, Learner, Program, actVarKeys, actVarVals,
        mutateParamKeys, mutateParamVals, doMemory, memType, operationSet):
    # change functions as needed
    if doMemory:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram11.execute_mem
            trainer.functionsDict["Program"]["execute"] = "mem"
            trainer.nOperations = 7
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "full":
            Program.execute = ConfProgram11.execute_mem_full
            trainer.functionsDict["Program"]["execute"] = "mem_full"
            trainer.nOperations = 10
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "robo":
            Program.execute = ConfProgram11.execute_mem_robo
            trainer.functionsDict["Program"]["execute"] = "mem_robo"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "custom":
            Program.execute = ConfProgram11.execute_mem_custom
            trainer.functionsDict["Program"]["execute"] = "mem_custom"
            trainer.nOperations = 18
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        # select appropriate memory write function
        if memType == "cauchy1":
            Program.memWriteProbFunc = ConfProgram11.memWriteProb_cauchy1
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchy1"
        elif memType == "cauchyHalf":
            Program.memWriteProbFunc = ConfProgram11.memWriteProb_cauchyHalf
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchyHalf"
        else:
            Program.memWriteProbFunc = ConfProgram11.memWriteProb_def
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "def"

        # change bid function to accomodate additional parameters needed for memory
        Learner.bid = ConfLearner11.bid_mem
        trainer.functionsDict["Learner"]["bid"] = "mem"

        # trainer needs to have memory
        trainer.memMatrix = np.zeros(shape=trainer.memMatrixShape)
        # agents need access to memory too, and to pass through act
        actVarKeys += ["memMatrix"]
        actVarVals += [trainer.memMatrix]

    else:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram11.execute_def
            trainer.functionsDict["Program"]["execute"] = "def"
            trainer.nOperations = 5
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG"]
        elif operationSet == "full":
            Program.execute = ConfProgram11.execute_full
            trainer.functionsDict["Program"]["execute"] = "full"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP"]
        elif operationSet == "robo":
            Program.execute = ConfProgram11.execute_robo
            trainer.functionsDict["Program"]["execute"] = "robo"
            trainer.nOperations = 6
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS"]
        elif operationSet == "custom":
            Program.execute = ConfProgram11.execute_custom
            trainer.functionsDict["Program"]["execute"] = "custom"
            trainer.nOperations = 16
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS"]


        Learner.bid = ConfLearner11.bid_def
        trainer.functionsDict["Learner"]["bid"] = "def"

    mutateParamKeys += ["nOperations"]
    mutateParamVals += [trainer.nOperations]

"""
Make the appropriate changes needed to be able to use real actions.
"""
def _configureRealAction11(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory):
    # change functions as needed
    ActionObject.__init__ = ConfActionObject11.init_real
    trainer.functionsDict["ActionObject"]["init"] = "real"
    ActionObject.getAction = ConfActionObject11.getAction_real
    trainer.functionsDict["ActionObject"]["getAction"] = "real"
    if doMemory:
        ActionObject.getRealAction = ConfActionObject11.getRealAction_real_mem
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real_mem"
    else:
        ActionObject.getRealAction = ConfActionObject11.getRealAction_real
        trainer.functionsDict["ActionObject"]["getRealAction"] = "real"
    ActionObject.mutate = ConfActionObject11.mutate_real
    trainer.functionsDict["ActionObject"]["mutate"] = "real"

    # mutateParams needs to have lengths of actions and act program
    mutateParamKeys += ["actionLengths", "initMaxActProgSize", "nActRegisters"]
    mutateParamVals += [trainer.actionLengths, trainer.initMaxActProgSize, trainer.nActRegisters]

"""
Switch to learner traversal.
"""
def _configureLearnerTraversal11(trainer, Agent, Team, actVarKeys, actVarVals):
    Team.act = ConfTeam11.act_learnerTrav
    trainer.functionsDict["Team"]["act"] = "learnerTrav"

def configure2(trainer, Trainer, Agent, Team, Learner, MemoryObject, Program,
        doMemory, memType, doReal, operationSet, traversal):

    # keys and values used in key value pairs for suplementary function args
    # for mutation and creation
    mutateParamKeys = ["generation", "maxTeamSize", "pLrnDel", "pLrnAdd", "pLrnMut",
        "pProgMut", "pMemMut", "pMemAtom", "pInstDel", "pInstAdd", "pInstSwp", "pInstMut",
        "memoryCodes", "nDestinations", "inputSize", "initMaxProgSize",
        "rampantGen", "rampantMin", "rampantMax", "idCountTeam", "idCountLearner", "idCountProgram"]
    mutateParamVals = [trainer.generation, trainer.maxTeamSize, trainer.pLrnDel, trainer.pLrnAdd, trainer.pLrnMut,
        trainer.pProgMut, trainer.pMemMut, trainer.pMemAtom, trainer.pInstDel, trainer.pInstAdd, trainer.pInstSwp, trainer.pInstMut,
        trainer.memoryCodes, trainer.nRegisters, trainer.inputSize, trainer.initMaxProgSize,
        trainer.rampancy[0], trainer.rampancy[1], trainer.rampancy[2], 0, 0, 0]

    # additional stuff for act, like memory matrix possible
    actVarKeys = ["frameNum"]
    actVarVals = [0]

    # before doing any special configuration, set all methods to defaults
    _configureDefaults2(trainer, Trainer, Agent, Team, Learner, MemoryObject, Program)

    # configure Program execution stuff, affected by memory and operations set
    _configureProgram2(trainer, Learner, Program, actVarKeys, actVarVals, mutateParamKeys, mutateParamVals, doMemory, memType, operationSet)

    # configure stuff for using real valued actions
    if doReal:  _configureRealMemory2(trainer, MemoryObject, mutateParamKeys, mutateParamVals, doMemory)

    # do learner traversal
    if traversal == "learner": _configureLearnerTraversal2(trainer, Agent, Team, actVarKeys, actVarVals)

    trainer.mutateParams = dict(zip(mutateParamKeys, mutateParamVals))
    trainer.actVars = dict(zip(actVarKeys, actVarVals))

"""
For each class in TPG, sets the functions to their defaults.
"""
def _configureDefaults2(trainer, Trainer, Agent, Team, Learner, MemoryObject, Program):
    # set trainer functions
    # TODO: add learner configurable

    # set agent functions
    Agent.__init__ = ConfAgent2.init_def
    Agent.image = ConfAgent2.image_def
    Agent.reward = ConfAgent2.reward_def
    Agent.taskDone = ConfAgent2.taskDone_def
    Agent.saveToFile = ConfAgent2.saveToFile_def

    # set team functions
    Team.__init__ = ConfTeam2.init_def
    Team.image = ConfTeam2.image_def
    Team.addLearner = ConfTeam2.addLearner_def
    Team.removeLearner = ConfTeam2.removeLearner_def
    Team.removeLearners = ConfTeam2.removeLearners_def
    Team.numAtomicMemories = ConfTeam2.numAtomicMemories_def
    Team.mutate = ConfTeam2.mutate_def
    Team.clone = ConfTeam2.clone_def

    # set learner functions
    Learner.__init__ = ConfLearner2.init_def
    Learner.bid = ConfLearner2.bid_def
    Learner.getImage = ConfLearner2.getImage_def
    Learner.getMemoryTeam = ConfLearner2.getMemoryTeam_def
    Learner.isMemoryAtomic = ConfLearner2.isMemoryAtomic_def
    Learner.mutate = ConfLearner2.mutate_def
    Learner.clone = ConfLearner2.clone_def

    # set program functions
    Program.__init__ = ConfProgram2.init_def
    Program.execute = ConfProgram2.execute_def
    Program.mutate = ConfProgram2.mutate_def
    Program.memWriteProbFunc = ConfProgram2.memWriteProb_def

    # let trainer know what functions are set for each one
    
    trainer.functionsDict["Agent"] = {
        "init": "def",
        "image": "def",
        "reward": "def",
        "taskDone": "def",
        "saveToFile": "def"
    }
    trainer.functionsDict["Team"] = {
        "init": "def",
        "image": "def",
        "addLearner": "def",
        "removeLearner": "def",
        "removeLearners": "def",
        "numAtomicMemories": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["Learner"] = {
        "init": "def",
        "bid": "def",
        "getImage": "def",
        "getMemoryTeam": "def",
        "isMemoryAtomic": "def",
        "mutate": "def",
        "clone": "def"
    }
    trainer.functionsDict["MemoryObject"] = {
        "init": "def",
        "getImage": "def",
        "getRealImage": "None",
        "isAtomic": "def",
        "mutate": "def"
    }
    trainer.functionsDict["Program"] = {
        "init": "def",
        "execute": "def",
        "mutate": "def",
        "memWriteProbFunc": "def"
    }

"""
Decides the operations and functions to be used in program execution.
"""
def _configureProgram2(trainer, Learner, Program, actVarKeys, actVarVals,
        mutateParamKeys, mutateParamVals, doMemory, memType, operationSet):
    # change functions as needed
    if doMemory:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram2.execute_mem
            trainer.functionsDict["Program"]["execute"] = "mem"
            trainer.nOperations = 7
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "full":
            Program.execute = ConfProgram2.execute_mem_full
            trainer.functionsDict["Program"]["execute"] = "mem_full"
            trainer.nOperations = 10
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "robo":
            Program.execute = ConfProgram2.execute_mem_robo
            trainer.functionsDict["Program"]["execute"] = "mem_robo"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "MEM_READ", "MEM_WRITE"]
        elif operationSet == "custom":
            Program.execute = ConfProgram2.execute_mem_custom
            trainer.functionsDict["Program"]["execute"] = "mem_custom"
            trainer.nOperations = 18
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

        # select appropriate memory write function
        if memType == "cauchy1":
            Program.memWriteProbFunc = ConfProgram2.memWriteProb_cauchy1
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchy1"
        elif memType == "cauchyHalf":
            Program.memWriteProbFunc = ConfProgram2.memWriteProb_cauchyHalf
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchyHalf"
        else:
            Program.memWriteProbFunc = ConfProgram2.memWriteProb_def
            trainer.functionsDict["Program"]["memWriteProbFunc"] = "def"

        # change bid function to accomodate additional parameters needed for memory
        Learner.bid = ConfLearner2.bid_mem
        trainer.functionsDict["Learner"]["bid"] = "mem"

        # trainer needs to have memory
        trainer.memMatrix = np.zeros(shape=trainer.memMatrixShape)
        # agents need access to memory too, and to pass through act
        actVarKeys += ["memMatrix"]
        actVarVals += [trainer.memMatrix]

    else:
        # default (reduced) or full operation set
        if operationSet == "def":
            Program.execute = ConfProgram2.execute_def
            trainer.functionsDict["Program"]["execute"] = "def"
            trainer.nOperations = 5
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG"]
        elif operationSet == "full":
            Program.execute = ConfProgram2.execute_full
            trainer.functionsDict["Program"]["execute"] = "full"
            trainer.nOperations = 8
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP"]
        elif operationSet == "robo":
            Program.execute = ConfProgram2.execute_robo
            trainer.functionsDict["Program"]["execute"] = "robo"
            trainer.nOperations = 6
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS"]
        elif operationSet == "custom":
            Program.execute = ConfProgram2.execute_custom
            trainer.functionsDict["Program"]["execute"] = "custom"
            trainer.nOperations = 16
            trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS"]


        Learner.bid = ConfLearner2.bid_def
        trainer.functionsDict["Learner"]["bid"] = "def"

    mutateParamKeys += ["nOperations"]
    mutateParamVals += [trainer.nOperations]

"""
Make the appropriate changes needed to be able to use real actions.
"""
def _configureRealMemory2(trainer, MemoryObject, mutateParamKeys, mutateParamVals, doMemory):
    # change functions as needed
    trainer.functionsDict["MemoryObject"]["init"] = "real"
    trainer.functionsDict["MemoryObject"]["getImage"] = "real"
    if doMemory:
        trainer.functionsDict["MemoryObject"]["getRealMemory"] = "real_mem"
    else:
        trainer.functionsDict["MemoryObject"]["getRealMemory"] = "real"
    trainer.functionsDict["MemoryObject"]["mutate"] = "real"

    # mutateParams needs to have lengths of actions and act program
    mutateParamKeys += ["memoryLengths", "initMaxActProgSize", "nMemRegisters"]
    mutateParamVals += [trainer.memoryLengths, trainer.initMaxActProgSize, trainer.nMemRegisters]

"""
Switch to learner traversal.
"""
def _configureLearnerTraversal2(trainer, Agent, Team, actVarKeys, actVarVals):
    Team.image = ConfTeam2.image_learnerTrav
    trainer.functionsDict["Team"]["image"] = "learnerTrav"
