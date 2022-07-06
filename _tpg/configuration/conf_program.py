import numpy as np
from numpy import pi, inf, NINF, float64, finfo
from numpy.random import rand
import math
from math import isnan, cos, log, exp, sin, sqrt, pow, tanh
import random
from _tpg.utils import flip
import uuid
import copy

"""
A program that is executed to help obtain the bid for a learner.
"""
class ConfProgram:

    def init_def(self, 
        instructions=None, 
        maxProgramLength=128, 
        nOperations=5,
        nDestinations=8,
        inputSize=30720,
        initParams=None
    ):
       
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
    def execute_def(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
            except Exception:  pass

            if math.isnan(regs[dest]):  regs[dest] = 0
            elif regs[dest] == np.inf:  regs[dest] = np.finfo(np.float64).max
            elif regs[dest] == np.NINF: regs[dest] = np.finfo(np.float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 6:
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
            except Exception: pass

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_full(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_full(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)
                elif op == 8:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 9:
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
    Executes the program which returns a single final value.
    """
    def execute_custom(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4:           regs[dest] = x**y
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

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_custom(inpt, regs, modes, ops, dsts, srcs,  memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4:           regs[dest] = x**y
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
    Executes the program which returns a single final value.
    """
    def execute_robo(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
            except Exception:  pass

                
            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_mem_robo(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 7:
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
    Returns probability of write at given index using default distribution.
    """
    def memWriteProb_def(i):
        return 0.25 - (0.01*i)**2

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1.
    """
    def memWriteProb_cauchy1(i):
        return 1/(pi*(i**2+1))

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1/2.
    """
    def memWriteProb_cauchyHalf(i):
        return 0.25/(0.5*pi*(i**2+0.25))

    """
    Mutates the program, by performing some operations on the instructions.
    """
    def mutate_def(self, mutateParams):
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
    Potentially modifies the instructions in a few ways.
    """
    def mutateInstructions_def(self, mutateParams):

        changed = False

        while not changed:
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,
                                    random.randint(0, len(self.instructions)-1),
                                    0)

                changed = True

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

                changed = True

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(mutateParams["pInstSwp"]):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)

                # do swap
                tmp = np.array(self.instructions[idx1])
                self.instructions[idx1] = np.array(self.instructions[idx2])
                self.instructions[idx2] = tmp

                changed = True

            # maybe add instruction
            if flip(mutateParams["pInstAdd"]):
                # insert new random instruction
                self.instructions = np.insert(self.instructions,
                        random.randint(0,len(self.instructions)),
                            (random.randint(0,1),
                            random.randint(0, mutateParams["nOperations"]-1),
                            random.randint(0, mutateParams["nDestinations"]-1),
                            random.randint(0, mutateParams["inputSize"]-1)),0)

                changed = True

class ConfProgram1:

    def init_def(self, instructions=None, maxProgramLength=128, nOperations=5,
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
    def execute_def(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
            except Exception:  pass

            if math.isnan(regs[dest]):  regs[dest] = 0
            elif regs[dest] == np.inf:  regs[dest] = np.finfo(np.float64).max
            elif regs[dest] == np.NINF: regs[dest] = np.finfo(np.float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 6:
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
            except Exception: pass

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_full(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_full(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)
                elif op == 8:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 9:
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
    Executes the program which returns a single final value.
    """
    def execute_custom(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4:           regs[dest] = x**y
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

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_custom(inpt, regs, modes, ops, dsts, srcs,  memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4:           regs[dest] = x**y
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
    Executes the program which returns a single final value.
    """
    def execute_robo(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
            except Exception:  pass

                
            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_mem_robo(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 7:
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
    Returns probability of write at given index using default distribution.
    """
    def memWriteProb_def(i):
        return 0.25 - (0.01*i)**2

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1.
    """
    def memWriteProb_cauchy1(i):
        return 1/(pi*(i**2+1))

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1/2.
    """
    def memWriteProb_cauchyHalf(i):
        return 0.25/(0.5*pi*(i**2+0.25))

    """
    Mutates the program, by performing some operations on the instructions.
    """
    def mutate_def(self, mutateParams):
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
    Potentially modifies the instructions in a few ways.
    """
    def mutateInstructions_def(self, mutateParams):

        changed = False

        while not changed:
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,
                                    random.randint(0, len(self.instructions)-1),
                                    0)

                changed = True

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

                changed = True

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(mutateParams["pInstSwp"]):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)

                # do swap
                tmp = np.array(self.instructions[idx1])
                self.instructions[idx1] = np.array(self.instructions[idx2])
                self.instructions[idx2] = tmp

                changed = True

            # maybe add instruction
            if flip(mutateParams["pInstAdd"]):
                # insert new random instruction
                self.instructions = np.insert(self.instructions,
                        random.randint(0,len(self.instructions)),
                            (random.randint(0,1),
                            random.randint(0, mutateParams["nOperations"]-1),
                            random.randint(0, mutateParams["nDestinations"]-1),
                            random.randint(0, mutateParams["inputSize"]-1)),0)

                changed = True

class ConfProgram11:

    def init_def(self, instructions=None, maxProgramLength=128, nOperations=5,
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
    def execute_def(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
            except Exception:  pass

            if math.isnan(regs[dest]):  regs[dest] = 0
            elif regs[dest] == np.inf:  regs[dest] = np.finfo(np.float64).max
            elif regs[dest] == np.NINF: regs[dest] = np.finfo(np.float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 6:
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
            except Exception: pass

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_full(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_full(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)
                elif op == 8:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 9:
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
    Executes the program which returns a single final value.
    """
    def execute_custom(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4:           regs[dest] = x**y
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

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_custom(inpt, regs, modes, ops, dsts, srcs,  memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4:           regs[dest] = x**y
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
    Executes the program which returns a single final value.
    """
    def execute_robo(inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
            except Exception:  pass

                
            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_mem_robo(inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 7:
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
    Returns probability of write at given index using default distribution.
    """
    def memWriteProb_def(i):
        return 0.25 - (0.01*i)**2

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1.
    """
    def memWriteProb_cauchy1(i):
        return 1/(pi*(i**2+1))

    """
    Returns probability of write at given index using cauchy distribution with
    lambda = 1/2.
    """
    def memWriteProb_cauchyHalf(i):
        return 0.25/(0.5*pi*(i**2+0.25))

    """
    Mutates the program, by performing some operations on the instructions.
    """
    def mutate_def(self, mutateParams):
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
    Potentially modifies the instructions in a few ways.
    """
    def mutateInstructions_def(self, mutateParams):

        changed = False

        while not changed:
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,
                                    random.randint(0, len(self.instructions)-1),
                                    0)

                changed = True

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

                changed = True

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(mutateParams["pInstSwp"]):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)

                # do swap
                tmp = np.array(self.instructions[idx1])
                self.instructions[idx1] = np.array(self.instructions[idx2])
                self.instructions[idx2] = tmp

                changed = True

            # maybe add instruction
            if flip(mutateParams["pInstAdd"]):
                # insert new random instruction
                self.instructions = np.insert(self.instructions,
                        random.randint(0,len(self.instructions)),
                            (random.randint(0,1),
                            random.randint(0, mutateParams["nOperations"]-1),
                            random.randint(0, mutateParams["nDestinations"]-1),
                            random.randint(0, mutateParams["inputSize"]-1)),0)

                changed = True

class ConfProgram2:

    def init_def(self, instructions=None, maxProgramLength=128, nOperations=5,
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


    def execute_def(act, inpt, regs, modes, ops, dsts, srcs):
        # inpt+=[act]
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]


            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
            except Exception:  pass

            if math.isnan(regs[dest]):  regs[dest] = 0
            elif regs[dest] == np.inf:  regs[dest] = np.finfo(np.float64).max
            elif regs[dest] == np.NINF: regs[dest] = np.finfo(np.float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem(act, inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 6:
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
            except Exception: pass

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value.
    """
    def execute_full(act, inpt, regs, modes, ops, dsts, srcs):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_full(act, inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if op == 0:             regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*2
                elif op == 3:           regs[dest] = x/2
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6 and y > 0: regs[dest] = log(y)
                elif op == 7:           regs[dest] = exp(y)
                elif op == 8:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 9:
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
    Executes the program which returns a single final value.
    """
    def execute_custom(act, inpt, regs, modes, ops, dsts, srcs):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if   op == 0:           regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*y
                elif op == 3 and y != 0:regs[dest] = x/y
                elif op == 4:           regs[dest] = x**y
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

            except Exception:  pass


            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    """
    Executes the program which returns a single final value using shared memory.
    """
    def execute_mem_custom(act, inpt, regs, modes, ops, dsts, srcs,  memMatrix, memRows, memCols, memWriteProbFunc):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize

            # do an operation
            try:
                if   op == 0:           regs[dest] = x+y
                elif op == 1:           regs[dest] = x-y
                elif op == 2:           regs[dest] = x*y
                elif op == 3 and y != 0:regs[dest] = x/y
                elif op == 4:           regs[dest] = x**y
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

    def execute_robo(act, inpt, regs, modes, ops, dsts, srcs):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
            except Exception:  pass

                
            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    def execute_mem_robo(act, inpt, regs, modes, ops, dsts, srcs,
            memMatrix, memRows, memCols, memWriteProbFunc):
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[(srcs[i]+act)%regSize]
            else:               src = inpt[(srcs[i]+act)%inptLen]

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
                elif op == 4 and x < y: regs[dest] = x*(-1)
                elif op == 5:           regs[dest] = cos(y)
                elif op == 6:
                    index = srcs[i]
                    index %= (memRows*memCols)
                    row = int(index / memRows)
                    col = index % memCols
                    regs[dest] = memMatrix[row, col]
                elif op == 7:
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

    def memWriteProb_def(i):
        return 0.25 - (0.01*i)**2

    def memWriteProb_cauchy1(i):
        return 1/(pi*(i**2+1))

    def memWriteProb_cauchyHalf(i):
        return 0.25/(0.5*pi*(i**2+0.25))

    def mutate_def(self, mutateParams):
        # Make a copy of our original instructions
        original_instructions = copy.deepcopy(self.instructions)

        # Since we're mutating change our id
        self.id = uuid.uuid4()

        # While we haven't changed from our original instructions keep mutating
        while np.array_equal(self.instructions, original_instructions):
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,random.randint(0, len(self.instructions)-1),0)

                

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

    def mutateInstructions_def(self, mutateParams):

        changed = False

        while not changed:
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(mutateParams["pInstDel"]):
                # delete random row/instruction
                self.instructions = np.delete(self.instructions,
                                    random.randint(0, len(self.instructions)-1),
                                    0)

                changed = True

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

                changed = True

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(mutateParams["pInstSwp"]):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)

                # do swap
                tmp = np.array(self.instructions[idx1])
                self.instructions[idx1] = np.array(self.instructions[idx2])
                self.instructions[idx2] = tmp

                changed = True

            # maybe add instruction
            if flip(mutateParams["pInstAdd"]):
                # insert new random instruction
                self.instructions = np.insert(self.instructions,
                        random.randint(0,len(self.instructions)),
                            (random.randint(0,1),
                            random.randint(0, mutateParams["nOperations"]-1),
                            random.randint(0, mutateParams["nDestinations"]-1),
                            random.randint(0, mutateParams["inputSize"]-1)),0)

                changed = True
