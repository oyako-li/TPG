from numpy import pi, inf, NINF, float64, finfo
from numpy.random import rand
# from numba import njit
from math import sin, cos, tanh, log, sqrt, exp, pow, isnan
from _tpg.utils import flip, _Logger
import numpy as np
import random
import uuid

GAMMA = [',','+','-','*','**','/','//','%','<','<=','<<','>','>=','>>','&','|','^','~']

class _Program(_Logger):

    # you should inherit
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
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

    def __eq__(self, __o:object) -> bool:
        '''
        A program is equal to another object if that object:
            - is an instance of the program class
            - has identical instructions
        '''
        # The other object must be an instance of the Program class
        if not isinstance(__o, self.__class__): return False

        # Compare instructions
        return np.array_equal(self.instructions, __o.instructions)

    def __ne__(self, __o: object) -> bool:
        '''
        Negation of __eq__
        '''
        return not self.__eq__(__o)

    def memWriteProb(i):
        """
        Returns probability of write at given index using cauchy distribution with
        lambda = 1.
        """
        return 1/(pi*(i**2+1))
  
    def mutate(self, mutateParams):
        """
        Potentially modifies the instructions in a few ways.
        """
        # Make a copy of our original instructions
        original_instructions = np.array(self.instructions)

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

    @classmethod
    def execute(cls, inpt, regs, modes, ops, dsts, srcs):
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
                elif op == 4:           pass #regs[dest] = x**y
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
                else:
                    regs[dest]=np.nan
            
            except Warning as w:
                regs[dest] = 0

            except Exception as e:
                regs[dest] = 0

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

class Program(_Program):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)
  
    @classmethod
    def execute(cls,
        inpt,   # state: np.ndarray or MemoryObj
        regs,   # self.registers: np.ndarray
        modes,  # self.program.instructions[:,0]: [random.randint(0,1), ...]
        ops,    # self.program.instructions[:,1]: [random.randint(0, oppelation.rang-1), ...]
        dsts,   # self.program.instructions[:,2]: [random.randint(0, register.len-1), ...]
        srcs,   # self.program.instructions[:,3]: [random.randint(0, state.size-1), ...]
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
            dest = (dsts[i])%regSize

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
                else:
                    regs[dest]=np.nan
            
            except Warning as w:
                # cls.warning(f'warning:"{w} on {cls.__name__}"')
                regs[dest] = 0

            except Exception as e:
                # cls.warning(f'error:"{e} on {cls.__name__}"')
                regs[dest] = 0

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

class Program1(Program):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)
    def __init__(self, 
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

        self._id = uuid.uuid4()

    def mutate(self, mutateParams):
        """
        Potentially modifies the instructions in a few ways.
        """
        # Make a copy of our original instructions
        original_instructions = np.array(self.instructions)

        # Since we're mutating change our id
        self._id = uuid.uuid4()

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

    @classmethod
    def id(self):
        return str(self._id)

class Program1_3(Program1):
    """Activate sequence create program
    GAMMA = [',','+','-','*','**','/','//','%','<','<=','<<','>','>=','>>','&','|','^','~']
    Done:TODO: 経過時間で、思考ブレイク。-> emulator(Hippocampus)ネットワークはプリミティブ演算のみを使用することによって無限ループを解決。
    TODO: Hippocampusの参照・リライト
    """
    Qualia=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.memory_object import Qualia
            cls._instance = True
            cls.Qualia=Qualia
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, 
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

        self._id = uuid.uuid4()

    @classmethod
    def execute(cls,
            inpt,   
            regs,   
            modes,  
            ops,    
            dsts,   
            srcs,   
            hippocampus 
        ):
        """ calicutate
        Attributes:
            inpt:   np.ndarray, MemoryObj, ActionObj or Qualia \
                = state
            regs:   np.ndarray[Qualia, ...] \
                = learner.registers  # signal context or Charge
            modes:  np.ndarray[random.randint(0,1), ...] \
                = learner.program.instructions[:,0]
            ops:    np.ndarray[random.randint(0, oppelation.rang-1), ...] \
                = learner.program.instructions[:,1] # 演算表現
            dsts:   np.ndarray[random.randint(0, register.len-1), ...] \
                = learner.program.instructions[:,2] # チューリング完全の書き込み場所。
            srcs:   np.ndarray[random.randint(0, state.size-1), ...] \
                = learner.program.instructions[:,3]
            hippocampus:    actVars['hippocampus'] \
                = agent.actVars['hippocampus'] # sharing short memory
        
        Return:
            None
                ref learner.register
        """
        inpt = cls.Qualia(inpt)
        regSize = len(regs)
        inptLen = len(inpt)
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:   src = regs[srcs[i]%regSize]
            else:               src = inpt[srcs[i]%inptLen]

            # get data for operation
            op = ops[i]
            x = regs[dsts[i]]
            y = cls.Qualia(src)
            dest = (dsts[i])%regSize

            # do an operation
            try:
                if   op == 0  :           regs[dest] = x+y
                elif op == 1  :           regs[dest] = x-y
                elif op == 2  :           regs[dest] = x*y
                elif op == 4  :           regs[dest] = x**y
                elif op == 5  and y != 0: regs[dest] = x/y
                elif op == 6  and y != 0: regs[dest] = x//y
                elif op == 7  :           regs[dest] = x%y
                elif op == 8  and x < y : regs[dest] = x*(-1)
                elif op == 9  and x <= y: regs[dest] = x*(-1)
                elif op == 10 :           regs[dest] = x<<y
                elif op == 11 and x > y : regs[dest] = x*(-1)
                elif op == 12 and x >= y: regs[dest] = x*(-1)
                elif op == 13 :           regs[dest] = x>>y
                elif op == 14 :           regs[dest] = x&y
                elif op == 15 :           regs[dest] = x|y
                elif op == 16 :           regs[dest] = x^y
                elif op == 17 :           regs[dest] = ~y
                # 無限再帰の可能性あり。 経過時間でブレイク　|　減衰　→ 別ネットワークの使用。そちらは、再帰性がなくても機能するプリミティブな処理だけで実装する
                # 再評価対象位置の設定はCerebellumのhippocampus.chにおいて設計される。
                # Cerebellumの出力チャンネルは拡張できるようにしたい。
                # 記憶アクセス
                elif op == 18 :           regs[dest] = hippocampus.suggestion(y)
                elif op == 19 :           regs[dest] = hippocampus.read(x)
                elif op == 20 :           hippocampus.write(regs[dest])
                else:                     regs[dest] = np.nan

            except Warning as w:
                regs[dest] = finfo(float64).min

            except Exception as e:
                regs[dest] = finfo(float64).min

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

    @property
    def id(self):
        return str(self._id)

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
                        
            except Warning as w:
                print(w, ' on calculation')
                regs[dest] = finfo(float64).min

            except Exception as e:
                print(f'{e} on program')
                regs[dest] = finfo(float64).min

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

class Program2_1(Program2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def execute(
        act: int or np.nan,
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
            y = src if src is not np.nan else 1
            act = act if act is not np.nan else -1
            try:
                dest = (dsts[i]+act)%regSize
            except Exception as e:
                print(e, f'{act}, {act.__class__}')
                dest = 0
            # assert (dsts[i]+act)%regSize is not None, f'{act}, {act.__class__}'

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
            
            except Warning as w:
                print(w, ' on calculation')
                regs[dest] = finfo(float64).min

            except Exception as e:
                print(f'{e} on program')
                regs[dest] = finfo(float64).min

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

class Program2_3(Program2):
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = True
        return super().__new__(cls, *args, **kwargs)

    def execute(
        # act: int or np.nan,
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
            y = src if src is not np.nan else 1
            # act = act if act is not np.nan else -1
            dest = (dsts[i])%regSize

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
            
            except Warning as w:
                regs[dest] = finfo(float64).min

            except Exception as e:
                regs[dest] = finfo(float64).min

            if isnan(regs[dest]):       regs[dest] = 0
            elif regs[dest] == inf:     regs[dest] = finfo(float64).max
            elif regs[dest] == NINF:    regs[dest] = finfo(float64).min

class Opelation(_Logger):
    Qualia=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.memory_object import Qualia
            cls._instance = True
            cls.Qualia=Qualia
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def caliculate(cls, _qualia, _opelation) -> list:
        # assert isinstance(_qualia[0], cls.Qualia) and isinstance(_opelation[0], cls.Qualia)

        regs = list(_qualia[0])
        
        for i, y in enumerate(_qualia[1:]):
            x = regs[-1]    
            if op:=_opelation[i]:
            # do an operation
                try:
                    if   op == 0  :           regs[-1] = x+y
                    elif op == 1  :           regs[-1] = x-y
                    elif op == 2  :           regs[-1] = x*y
                    elif op == 3  :           regs[-1] = x**y
                    elif op == 4  and y != 0: regs[-1] = x/y
                    elif op == 5  and y != 0: regs[-1] = x//y
                    elif op == 6  :           regs[-1] = x%y
                    elif op == 7  and x < y : regs[-1] = x*(-1)
                    elif op == 8  and x <= y: regs[-1] = x*(-1)
                    elif op == 9  :           regs[-1] = x<<y
                    elif op == 10 and x > y : regs[-1] = x*(-1)
                    elif op == 11 and x >= y: regs[-1] = x*(-1)
                    elif op == 12 :           regs[-1] = x>>y
                    elif op == 13 :           regs[-1] = x&y
                    elif op == 14 :           regs[-1] = x|y
                    elif op == 15 :           regs[-1] = x^y
                    elif op == 16 :           regs[-1] = ~y
                    # 無限再帰の可能性あり。 経過時間でブレイク　|　減衰　→ 別ネットワークの使用。そちらは、再帰性がなくても機能するプリミティブな処理だけで実装する
                    # 再評価対象位置の設定はCerebellumのhippocampus.chにおいて設計される。
                    # Cerebellumの出力チャンネルは拡張できるようにしたい。
                    # 記憶アクセス
                    # elif op == 18 :           regs[i] = hippocampus.suggestion(y)
                    # elif op == 19 :           regs[i] = hippocampus.read(x)
                    # elif op == 20 :           hippocampus.write(regs[i])
                    elif op == 17 :           regs[-1] = np.nan
                    else:                     regs.append(y)

                except Warning as w:
                    regs[-1] = finfo(float64).min

                except Exception as e:
                    regs[-1] = finfo(float64).min
            
            else: regs.append(y)

        return regs

class Activator(Opelation):
    """ 行動演算オペレーション
    
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.memory_object import Qualia
            cls._instance = True
            cls.Qualia=Qualia
        return super().__new__(cls, *args, **kwargs)

class Recollector(Opelation):
    """ 記憶演算オペレーション
    
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            from _tpg.memory_object import Qualia
            cls._instance = True
            cls.Qualia=Qualia
        return super().__new__(cls, *args, **kwargs)
