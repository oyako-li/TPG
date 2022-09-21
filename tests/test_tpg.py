import unittest
import gym

class TPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import TPG
        self.TPG = TPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.actions = self.env.action_space.n

    def test_init(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)

    # @unittest.skip('next test case')
    def test_episode(self):
        '''test episode'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        score = tpg.episode()
        self.assertIsNotNone(score)
        # agents = tpg.getAgents()

    # @unittest.skip('next test case')
    def test_generations(self):
        '''test generation'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        score = tpg.generation()
        self.assertIsNotNone(score)

    # @unittest.skip('next test case')
    def test_growing(self):
        '''test growing'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        filename = tpg.growing(_test=True)
        self.assertIsNotNone(filename)

class MHTPGTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        from _tpg.tpg import MHTPG
        self.TPG = MHTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.actions = self.env.action_space.n

    def test_init_(self):
        '''test init'''
        from _tpg.trainer import Trainer1
        tpg = self.TPG()
        self.assertIsInstance(tpg.trainer, Trainer1)

    def test_growing(self):
        '''test growing'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        filename = tpg.growing(_test=True)
        self.assertIsNotNone(filename)

if __name__ == '__main__':
    unittest.main()
