from dis import Instruction
import unittest
import numpy as np

class _LearnerTest(unittest.TestCase):
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

        state = self.env.observation_space.sample()
        program = self.Program()
        register = np.array([1]*10)
        pre_register = np.array([1]*10)
        self.assertEqual(np.sum(register-pre_register),0)
        self.Program.execute(
            state,
            register,
            program.instructions[:,0],
            program.instructions[:,1],
            program.instructions[:,2],
            program.instructions[:,3]
        )
        self.assertNotEqual(np.sum(register-pre_register), 0)


if __name__ == '__main__':
    unittest.main()