import unittest
import numpy as np
import copy

class _ProgramTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.program import _Program
        import gym
        self.Program = _Program
        self.env = gym.make("BreakoutNoFrameskip-v4")

    def test_init(self):
        '''test null program creation'''
        program = self.Program()
        self.assertIsNotNone(program.instructions)
        self.assertIsNotNone(program.id)

    def test_execute(self):
        '''test execution'''
        state = self.env.observation_space.sample()
        program = self.Program()
        register = np.array([1]*10)
        pre_register = np.array([1]*10)
        memMatrix = np.zeros(shape=(100,8))
        memRows = memMatrix[0]
        memCols = memMatrix[1]

        self.assertCountEqual(register, pre_register)
        self.Program.execute(
            state,
            register,
            program.instructions[:,0],
            program.instructions[:,1],
            program.instructions[:,2],
            program.instructions[:,3],
            memMatrix,                  # actVars["memMatrix"],
            memRows,                    # actVars["memMatrix"].shape[0],
            memCols,                    # actVars["memMatrix"].shape[1],
            program.memWriteProb
        )
        with self.assertRaises(AssertionError):
            self.assertCountEqual(register, pre_register)

    def test_mutate(self):
        '''test mutation '''
        mutate_params = {
            "pInstDel": 0.5,
            "pInstMut": 1.0,
            "pInstSwp": 0.4,
            "pInstAdd": 0.4,
            "nOperations": 16,
            "nDestinations": 8,
            "inputSize": 33600,
        }
        program = self.Program()
        ord_instractions = copy.deepcopy(program.instructions)
        program.mutate(mutate_params)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(ord_instractions, program.instructions)


    def test_eq_ne_(self):
        '''test equal and not equal '''
        mutate_params = {
            "pInstDel": 0.5,
            "pInstMut": 1.0,
            "pInstSwp": 0.4,
            "pInstAdd": 0.4,
            "nOperations": 16,
            "nDestinations": 8,
            "inputSize": 33600,
        }
        program = self.Program()
        program1 = self.Program()
        pre_program = copy.deepcopy(program)
        program.mutate(mutate_params)
        
        
        self.assertEqual(program, program)
        self.assertNotEqual(pre_program, program)
        self.assertNotEqual(program, program1)

if __name__ == '__main__':
    unittest.main()