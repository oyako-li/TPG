import random
import numpy as np
# from numba import njit
import math
import copy
from _tpg.utils import flip
import uuid
import logging

logger = logging.getLogger(__name__)


"""
A program that is executed to help obtain the bid for a learner.
"""
class Program:

    def __init__(self, instructions=None, maxProgramLength=128, nOperations=5,
            nDestinations=8, inputSize=30720, initParams=None):

        if instructions is not None: # copy from existing
            self.instructions = np.array(instructions, dtype=np.int32)
        else: # create random new
            self.instructions = np.array([
                (
                    random.randint(0,1),
                    random.randint(0, nOperations-1),
                    random.randint(0, nDestinations-1),
                    random.randint(0, inputSize-1)
                ) for _ in range(random.randint(1, maxProgramLength))], dtype=np.int32)

        self.id = uuid.uuid4()

    '''
    A program is equal to another object if that object:
        - is an instance of the program class
        - has identical instructions
    '''
    def __eq__(self, __o:object) -> bool:

        # The other object must be an instance of the Program class
        if not isinstance(__o, Program): return False

        # Compare instructions
        return np.array_equal(self.instructions, __o.instructions)

    '''
     Negation of __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    """
    Executes the program which returns a single final value.
    """
    def execute(
            inputState:np.ndarray,  # state
            registers:np.ndarray,   # self.registers
            modes:np.ndarray,       # self.program.instructions[:,0]
            operations:np.ndarray,  # self.program.instructions[:,1]
            dsts:np.ndarray,        # self.program.instructions[:,2]
            srcs:np.ndarray         # self.program.instructions[:,3]
        ):
        regSize = len(registers)
        inputLen = len(inputState)
        
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = registers[srcs[i]%regSize]
            else:               src = inputState[srcs[i]%inputLen]


            # get data for operation
            operation = operations[i]
            x = registers[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if operation == 0:              registers[dest] = x+y
                elif operation == 1:            registers[dest] = x-y
                elif operation == 2:            registers[dest] = x*2
                elif operation == 3:            registers[dest] = x/2
                elif operation == 4 and x < y:  registers[dest] = x*(-1)
            except Exception:  pass
                
            if math.isnan(registers[dest]): registers[dest] = 0
            elif registers[dest] == np.inf: registers[dest] = np.finfo(np.float64).max
            elif registers[dest] == np.NINF:registers[dest] = np.finfo(np.float64).min
            

    """
    Potentially modifies the instructions in a few ways.
    """
    def mutate(self, mutateParams):
        
        # Make a copy of our original instructions
        original_instructions = copy.deepcopy(self.instructions)

        # Since we're mutating change our id
        self.id = uuid.uuid4()

        # While we haven't changed from our original instructions keep mutating
        while np.array_equal(self.instructions, original_instructions):
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,
                                    random.randint(0, len(self.instructions)-1),
                                    0)

                

            # maybe mutate an instruction (flip a bit)
            if flip(mutateParams["pInstMut"]):
                # index of instruction and part of instruction
                idx1 = random.randint(0, len(self.instructions)-1)
                idx2 = random.randint(0,3)

                # change max value depending on part of instruction
                if idx2 == 0:
                    maxVal = 1
                elif idx2 == 1:
                    maxVal = mutateParams["nOperations"]-1
                elif idx2 == 2:
                    maxVal = mutateParams["nDestinations"]-1
                elif idx2 == 3:
                    maxVal = mutateParams["inputSize"]-1

                # change it
                self.instructions[idx1, idx2] = random.randint(0, maxVal)

                

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(mutateParams["pInstSwp"]):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)

                # do swap
                tmp = np.array(self.instructions[idx1])
                self.instructions[idx1] = np.array(self.instructions[idx2])
                self.instructions[idx2] = tmp

                

            # maybe add instruction
            if flip(mutateParams["pInstAdd"]):
                # insert new random instruction
                self.instructions = np.insert(self.instructions,
                        random.randint(0,len(self.instructions)),
                            (random.randint(0,1),
                            random.randint(0, mutateParams["nOperations"]-1),
                            random.randint(0, mutateParams["nDestinations"]-1),
                            random.randint(0, mutateParams["inputSize"]-1)),0)
            
            return self



    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_program import ConfProgram

        if functionsDict["init"] == "def":
            cls.__init__ = ConfProgram.init_def

        if functionsDict["execute"] == "def":
            cls.execute = ConfProgram.execute_def
        elif functionsDict["execute"] == "full":
            cls.execute = ConfProgram.execute_full
        elif functionsDict["execute"] == "custom":
            cls.execute = ConfProgram.execute_custom
        elif functionsDict["execute"] == "robo":
            cls.execute = ConfProgram.execute_robo
        elif functionsDict["execute"] == "mem":
            cls.execute = ConfProgram.execute_mem
        elif functionsDict["execute"] == "mem_full":
            cls.execute = ConfProgram.execute_mem_full
        elif functionsDict["execute"] == "mem_custom":
            cls.execute = ConfProgram.execute_mem_custom
        elif functionsDict["execute"] == "mem_robo":
            cls.execute = ConfProgram.execute_mem_robo

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfProgram.mutate_def
        
        if functionsDict["memWriteProbFunc"] == "def":
            cls.memWriteProbFunc = ConfProgram.memWriteProb_def
        elif functionsDict["memWriteProbFunc"] == "cauchy1":
            cls.memWriteProbFunc = ConfProgram.memWriteProb_cauchy1
        elif functionsDict["memWriteProbFunc"] == "cauchyHalf":
            cls.memWriteProbFunc = ConfProgram.memWriteProb_cauchyHalf

class Program1:

    def __init__(self, instructions=None, maxProgramLength=128, nOperations=5,
            nDestinations=8, inputSize=30720, initParams=None):
       
        if instructions is not None: # copy from existing
            self.instructions = np.array(instructions, dtype=np.int32)
        else: # create random new
            self.instructions = np.array([
                (
                    random.randint(0,1),
                    random.randint(0, nOperations-1),
                    random.randint(0, nDestinations-1),
                    random.randint(0, inputSize-1)
                ) for _ in range(random.randint(1, maxProgramLength))], dtype=np.int32)

        self.id = uuid.uuid4()

    """
    Executes the program which returns a single final value.
    """
    def execute(
            inputState:np.ndarray,  # state
            registers:np.ndarray,   # self.registers
            modes:np.ndarray,       # self.program.instructions[:,0]
            operations:np.ndarray,  # self.program.instructions[:,1]
            dsts:np.ndarray,        # self.program.instructions[:,2]
            srcs:np.ndarray         # self.program.instructions[:,3]
        ): pass
    
    """
    Potentially modifies the instructions in a few ways.
    """
    def mutate(self, mutateParams): pass


    '''
    A program is equal to another object if that object:
        - is an instance of the program class
        - has identical instructions
    '''
    def __eq__(self, __o:object) -> bool:

        # The other object must be an instance of the Program class
        if not isinstance(__o, Program1): return False

        # Compare instructions
        return np.array_equal(self.instructions, __o.instructions)

    '''
     Negation of __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_program import ConfProgram1

        if functionsDict["init"] == "def":
            cls.__init__ = ConfProgram1.init_def

        if functionsDict["execute"] == "def":
            cls.execute = ConfProgram1.execute_def
        elif functionsDict["execute"] == "full":
            cls.execute = ConfProgram1.execute_full
        elif functionsDict["execute"] == "custom":
            cls.execute = ConfProgram1.execute_custom
        elif functionsDict["execute"] == "robo":
            cls.execute = ConfProgram1.execute_robo
        elif functionsDict["execute"] == "mem":
            cls.execute = ConfProgram1.execute_mem
        elif functionsDict["execute"] == "mem_full":
            cls.execute = ConfProgram1.execute_mem_full
        elif functionsDict["execute"] == "mem_custom":
            cls.execute = ConfProgram1.execute_mem_custom
        elif functionsDict["execute"] == "mem_robo":
            cls.execute = ConfProgram1.execute_mem_robo

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfProgram1.mutate_def
        
        if functionsDict["memWriteProbFunc"] == "def":
            cls.memWriteProbFunc = ConfProgram1.memWriteProb_def
        elif functionsDict["memWriteProbFunc"] == "cauchy1":
            cls.memWriteProbFunc = ConfProgram1.memWriteProb_cauchy1
        elif functionsDict["memWriteProbFunc"] == "cauchyHalf":
            cls.memWriteProbFunc = ConfProgram1.memWriteProb_cauchyHalf

class Program2:

    def __init__(self, instructions=None, maxProgramLength=128, nOperations=5, nDestinations=8, inputSize=30720, initParams=None): pass

    """
    Executes the program which returns a single final value.
    """
    def execute(
            inputState:np.ndarray,  # state
            registers:np.ndarray,   # self.registers
            modes:np.ndarray,       # self.program.instructions[:,0]
            operations:np.ndarray,  # self.program.instructions[:,1]
            dsts:np.ndarray,        # self.program.instructions[:,2]
            srcs:np.ndarray         # self.program.instructions[:,3]
        ): pass
    
    """
    Potentially modifies the instructions in a few ways.
    """
    def mutate(self, mutateParams): pass


    '''
    A program is equal to another object if that object:
        - is an instance of the program class
        - has identical instructions
    '''
    def __eq__(self, __o:object) -> bool:

        # The other object must be an instance of the Program class
        if not isinstance(__o, Program2): return False

        # Compare instructions
        return np.array_equal(self.instructions, __o.instructions)

    '''
     Negation of __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


    """
    Ensures proper functions are used in this class as set up by configurer.
    """
    @classmethod
    def configFunctions(cls, functionsDict):
        from _tpg.configuration.conf_program import ConfProgram2

        if functionsDict["init"] == "def":
            cls.__init__ = ConfProgram2.init_def

        if functionsDict["execute"] == "def":
            cls.execute = ConfProgram2.execute_def
        elif functionsDict["execute"] == "full":
            cls.execute = ConfProgram2.execute_full
        elif functionsDict["execute"] == "custom":
            cls.execute = ConfProgram2.execute_custom
        elif functionsDict["execute"] == "robo":
            cls.execute = ConfProgram2.execute_robo
        elif functionsDict["execute"] == "mem":
            cls.execute = ConfProgram2.execute_mem
        elif functionsDict["execute"] == "mem_full":
            cls.execute = ConfProgram2.execute_mem_full
        elif functionsDict["execute"] == "mem_custom":
            cls.execute = ConfProgram2.execute_mem_custom
        elif functionsDict["execute"] == "mem_robo":
            cls.execute = ConfProgram2.execute_mem_robo

        if functionsDict["mutate"] == "def":
            cls.mutate = ConfProgram2.mutate_def
        
        if functionsDict["memWriteProbFunc"] == "def":
            cls.memWriteProbFunc = ConfProgram2.memWriteProb_def
        elif functionsDict["memWriteProbFunc"] == "cauchy1":
            cls.memWriteProbFunc = ConfProgram2.memWriteProb_cauchy1
        elif functionsDict["memWriteProbFunc"] == "cauchyHalf":
            cls.memWriteProbFunc = ConfProgram2.memWriteProb_cauchyHalf

