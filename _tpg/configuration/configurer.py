from _tpg.action_object import ActionObject3
# from _tpg.configuration.conf_emulator import ConfEmulator
import numpy as np



"""
For each class in TPG, sets the functions to their defaults.
"""

"""
Decides the operations and functions to be used in program execution.
"""

"""
Make the appropriate changes needed to be able to use real actions.
"""

"""
Switch to learner traversal.
"""

class Config:

    def importance(self):
        from _tpg.configuration.conf_agent import ConfAgent
        from _tpg.configuration.conf_team import ConfTeam
        from _tpg.configuration.conf_learner import ConfLearner
        from _tpg.configuration.conf_action_object import ConfActionObject
        from _tpg.configuration.conf_program import ConfProgram

        self.ConfAgent = ConfAgent
        self.ConfTeam = ConfTeam
        self.ConfLearner = ConfLearner
        self.ConfActionObject = ConfActionObject
        self.ConfProgram = ConfProgram

    def __init__(self) -> None:
        self.importance() 

    """
    Contains the ability to swap out different functions and add arbitrary variables
    to different classes willy nilly to support whatever functionality is desired.
    Such as Memory or real actions. Default configuration is no memory and discrete
    actions.
    """
    def configure(self, trainer, Agent, Team, Learner, ActionObject, Program,
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
        self.configureDefaults(trainer, Agent, Team, Learner, ActionObject, Program)

        # configure Program execution stuff, affected by memory and operations set
        self.configureProgram(trainer, Learner, Program, actVarKeys, actVarVals, mutateParamKeys, mutateParamVals, doMemory, memType, operationSet)

        # configure stuff for using real valued actions
        if doReal:  self.configureRealAction(trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory)

        # do learner traversal
        if traversal == "learner": self.configureLearnerTraversal(trainer, Agent, Team, actVarKeys, actVarVals)

        trainer.mutateParams = dict(zip(mutateParamKeys, mutateParamVals))
        trainer.actVars = dict(zip(actVarKeys, actVarVals))

    """
    For each class in TPG, sets the functions to their defaults.
    """
    def configureDefaults(self, trainer, Agent, Team, Learner, ActionObject, Program):
        # set trainer functions
        # TODO: add learner configurable

        # set agent functions
        Agent.__init__ = self.ConfAgent.init_def
        Agent.act = self.ConfAgent.act_def
        Agent.reward = self.ConfAgent.reward_def
        Agent.taskDone = self.ConfAgent.taskDone_def
        Agent.saveToFile = self.ConfAgent.saveToFile_def

        # set team functions
        Team.__init__ = self.ConfTeam.init_def
        Team.act = self.ConfTeam.act_def
        Team.addLearner = self.ConfTeam.addLearner_def
        Team.removeLearner = self.ConfTeam.removeLearner_def
        Team.removeLearners = self.ConfTeam.removeLearners_def
        Team.numAtomicActions = self.ConfTeam.numAtomicActions_def
        Team.mutate = self.ConfTeam.mutate_def
        Team.clone = self.ConfTeam.clone_def

        # set learner functions
        Learner.__init__ = self.ConfLearner.init_def
        Learner.bid = self.ConfLearner.bid_def
        Learner.getAction = self.ConfLearner.getAction_def
        Learner.getActionTeam = self.ConfLearner.getActionTeam_def
        Learner.isActionAtomic = self.ConfLearner.isActionAtomic_def
        Learner.mutate = self.ConfLearner.mutate_def
        Learner.clone = self.ConfLearner.clone_def


        # set action object functions
        ActionObject.__init__ = self.ConfActionObject.init_def
        ActionObject.getAction = self.ConfActionObject.getAction_def
        ActionObject.isAtomic = self.ConfActionObject.isAtomic_def
        ActionObject.mutate = self.ConfActionObject.mutate_def
        ActionObject._actions = trainer.actionCodes


        # set program functions
        Program.__init__ = self.ConfProgram.init_def
        Program.execute = self.ConfProgram.execute_def
        Program.mutate = self.ConfProgram.mutate_def
        Program.memWriteProbFunc = self.ConfProgram.memWriteProb_def

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
    def configureProgram(self, trainer, Learner, Program, actVarKeys, actVarVals,
            mutateParamKeys, mutateParamVals, doMemory, memType, operationSet):
        # change functions as needed
        if doMemory:
            # default (reduced) or full operation set
            if operationSet == "def":
                Program.execute = self.ConfProgram.execute_mem
                trainer.functionsDict["Program"]["execute"] = "mem"
                trainer.nOperations = 7
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "MEM_READ", "MEM_WRITE"]
            elif operationSet == "full":
                Program.execute = self.ConfProgram.execute_mem_full
                trainer.functionsDict["Program"]["execute"] = "mem_full"
                trainer.nOperations = 10
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP", "MEM_READ", "MEM_WRITE"]
            elif operationSet == "robo":
                Program.execute = self.ConfProgram.execute_mem_robo
                trainer.functionsDict["Program"]["execute"] = "mem_robo"
                trainer.nOperations = 8
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "MEM_READ", "MEM_WRITE"]
            elif operationSet == "custom":
                Program.execute = self.ConfProgram.execute_mem_custom
                trainer.functionsDict["Program"]["execute"] = "mem_custom"
                trainer.nOperations = 18
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS", "MEM_READ", "MEM_WRITE"]

            # select appropriate memory write function
            if memType == "cauchy1":
                Program.memWriteProbFunc = self.ConfProgram.memWriteProb_cauchy1
                trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchy1"
            elif memType == "cauchyHalf":
                Program.memWriteProbFunc = self.ConfProgram.memWriteProb_cauchyHalf
                trainer.functionsDict["Program"]["memWriteProbFunc"] = "cauchyHalf"
            else:
                Program.memWriteProbFunc = self.ConfProgram.memWriteProb_def
                trainer.functionsDict["Program"]["memWriteProbFunc"] = "def"

            # change bid function to accomodate additional parameters needed for memory
            Learner.bid = self.ConfLearner.bid_mem
            trainer.functionsDict["Learner"]["bid"] = "mem"

            # trainer needs to have memory
            trainer.memMatrix = np.zeros(shape=trainer.memMatrixShape)
            # agents need access to memory too, and to pass through act
            actVarKeys += ["memMatrix"]
            actVarVals += [trainer.memMatrix]

        else:
            # default (reduced) or full operation set
            if operationSet == "def":
                Program.execute = self.ConfProgram.execute_def
                trainer.functionsDict["Program"]["execute"] = "def"
                trainer.nOperations = 5
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG"]
            elif operationSet == "full":
                Program.execute = self.ConfProgram.execute_full
                trainer.functionsDict["Program"]["execute"] = "full"
                trainer.nOperations = 8
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS", "LOG", "EXP"]
            elif operationSet == "robo":
                Program.execute = self.ConfProgram.execute_robo
                trainer.functionsDict["Program"]["execute"] = "robo"
                trainer.nOperations = 6
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "NEG", "COS"]
            elif operationSet == "custom":
                Program.execute = self.ConfProgram.execute_custom
                trainer.functionsDict["Program"]["execute"] = "custom"
                trainer.nOperations = 16
                trainer.operations = ["ADD", "SUB", "MULT", "DIV", "POW", "NEG", "INV_NEG", "SIN", "COS", "TANH", "LN", "SQRT", "EXP", "POWY2", "POWY3", "ABS"]


            Learner.bid = self.ConfLearner.bid_def
            trainer.functionsDict["Learner"]["bid"] = "def"

        mutateParamKeys += ["nOperations"]
        mutateParamVals += [trainer.nOperations]

    """
    Make the appropriate changes needed to be able to use real actions.
    """
    def configureRealAction(self, trainer, ActionObject, mutateParamKeys, mutateParamVals, doMemory):
        # change functions as needed
        ActionObject.__init__ = self.ConfActionObject.init_real
        trainer.functionsDict["ActionObject"]["init"] = "real"
        ActionObject.getAction = self.ConfActionObject.getAction_real
        trainer.functionsDict["ActionObject"]["getAction"] = "real"
        if doMemory:
            ActionObject.getRealAction = self.ConfActionObject.getRealAction_real_mem
            trainer.functionsDict["ActionObject"]["getRealAction"] = "real_mem"
        else:
            ActionObject.getRealAction = self.ConfActionObject.getRealAction_real
            trainer.functionsDict["ActionObject"]["getRealAction"] = "real"
        ActionObject.mutate = self.ConfActionObject.mutate_real
        trainer.functionsDict["ActionObject"]["mutate"] = "real"

        # mutateParams needs to have lengths of actions and act program
        mutateParamKeys += ["actionLengths", "initMaxActProgSize", "nActRegisters"]
        mutateParamVals += [trainer.actionLengths, trainer.initMaxActProgSize, trainer.nActRegisters]

    """
    Switch to learner traversal.
    """
    def configureLearnerTraversal(self, trainer, Agent, Team, actVarKeys, actVarVals):
        Team.act = self.ConfTeam.act_learnerTrav
        trainer.functionsDict["Team"]["act"] = "learnerTrav"
