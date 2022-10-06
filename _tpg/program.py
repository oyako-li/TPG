import random
import numpy as np
from numpy import pi, inf, NINF, float64, finfo
from numpy.random import rand
# from numba import njit
from math import sin, cos, tanh, log, sqrt, exp, pow, isnan
import copy
from _tpg.utils import flip
import uuid

class _Program:
    _instance = None

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls)

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
            inpt:np.ndarray,        # state
            regs:np.ndarray,        # self.registers
            modes:np.ndarray,       # self.program.instructions[:,0]
            ops:np.ndarray,         # self.program.instructions[:,1]
            dsts:np.ndarray,        # self.program.instructions[:,2]
            srcs:np.ndarray,        # self.program.instructions[:,3]
            memMatrix, memRows, memCols, memWriteProbFunc
        ): 
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[srcs[i]%regSize]
            else:               src = inpt[srcs[i]%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*y
                elif op == 3 and y != 0:regs[dest] = x/y
                elif op == 4:           pass#regs[dest] = x**y
                elif op == 5 and x < y: regs[dest] = x*(-1)
                elif op == 6 and x > y: regs[dest] = x*(-1)
                elif op == 7:           regs[dest] = sin(y)
                elif op == 8:           regs[dest] = cos(y)
                elif op == 9:           regs[dest] = tanh(y)
                elif op == 10 and y > 0:regs[dest] = log(y)
                elif op == 11 and y > 0:regs[dest] = sqrt(y)
                elif op == 12:          regs[dest] = exp(y)
                elif op == 13:          regs[dest] = pow(y,2)
                elif op == 14:          regs[dest] = pow(y,3)
                elif op == 15:          regs[dest] = abs(y)
                elif op == 16:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 17:
                    # row offset (start from center, go to edges)
                    halfRows = int(memRows/2) # halfRows
                    for i in range(halfRows):
                        # probability to write (gets smaller as i increases)
                        # TODO: swap out write prob func by passing in an array of values for that row.
                        writeProb = memWriteProbFunc(i)
                        # column to maybe write corresponding value into
                        for col in range(memCols):
                            # try write to lower half
                            if rand(1)[0] < writeProb:
                                row = (halfRows - i) - 1
                                memMatrix[row,col] = regs[col]
                            # try write to upper half
                            if rand(1)[0] < writeProb:
                                row = halfRows + i
                                memMatrix[row,col] = regs[col]
            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    
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

        '''
    A program is equal to another object if that object:
        - is an instance of the program class
        - has identical instructions
    '''

    def __eq__(self, __o:object) -> bool:
        # The other object must be an instance of the Program class
        if not isinstance(__o, self.__class__): return False

        # Compare instructions
        return np.array_equal(self.instructions, __o.instructions)

    '''
     Negation of __eq__
    '''
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1.
    """
    def memWriteProb(i):
        return 1/(pi*(i**2+1))

class Program1(_Program):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

class Program2(_Program):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def execute(
        act: int,
        inpt: np.ndarray, 
        regs: np.ndarray, 
        modes: np.ndarray, 
        ops: np.ndarray, 
        dsts: np.ndarray, 
        srcs: np.ndarray, 
        memMatrix, memRows, memCols, memWriteProbFunc
    ):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[srcs[i]%regSize]
            else:               src = inpt[srcs[i]%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = (dsts[i]+act)%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*y
                elif op == 3 and y != 0:regs[dest] = x/y
                elif op == 4:           pass#regs[dest] = x**y
                elif op == 5 and x < y: regs[dest] = x*(-1)
                elif op == 6 and x > y: regs[dest] = x*(-1)
                elif op == 7:           regs[dest] = sin(y)
                elif op == 8:           regs[dest] = cos(y)
                elif op == 9:           regs[dest] = tanh(y)
                elif op == 10 and y > 0:regs[dest] = log(y)
                elif op == 11 and y > 0:regs[dest] = sqrt(y)
                elif op == 12:          regs[dest] = exp(y)
                elif op == 13:          regs[dest] = pow(y,2)
                elif op == 14:          regs[dest] = pow(y,3)
                elif op == 15:          regs[dest] = abs(y)
                elif op == 16:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 17:
                    # row offset (start from center, go to edges)
                    halfRows = int(memRows/2) # halfRows
                    for i in range(halfRows):
                        # probability to write (gets smaller as i increases)
                        # TODO: swap out write prob func by passing in an array of values for that row.
                        writeProb = memWriteProbFunc(i)
                        # column to maybe write corresponding value into
                        for col in range(memCols):
                            # try write to lower half
                            if rand(1)[0] < writeProb:
                                row = (halfRows - i) - 1
                                memMatrix[row,col] = regs[col]
                            # try write to upper half
                            if rand(1)[0] < writeProb:
                                row = halfRows + i
                                memMatrix[row,col] = regs[col]
            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min
