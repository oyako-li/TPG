import unittest
import gym
from _tpg.base_log import log_show

class TPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import _TPG
        self.TPG = _TPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.actions = self.env.action_space.n

    def test_init(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)

    @unittest.skip('next test case')
    def test_episode(self):
        '''test episode'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        _scores = {}
        _task = self.env.spec.id
        for _ in range(1):     
            _scores = tpg.episode()
        for i in _scores:               
            _scores[i]/=1
        for agent in tpg.trainer.getAgents(): 
            agent.reward(_scores[str(agent.team.id)],task=_task)
        tpg.trainer.evolve([_task])
        # agents = tpg.getAgents()

    @unittest.skip('next test case')
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
        filename = tpg.growing(_dir='test/')
        self.assertIsNotNone(filename)
        log_show(f'log/{filename}')

# @unittest.skip('before')
class MHTPGTest(TPGTest):

    def setUp(self) -> None:
        from _tpg.tpg import MHTPG
        self.TPG = MHTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.actions = self.env.action_space.n

    def test_init_(self):
        '''test init'''
        from _tpg.tpg import MHTPG
        self.assertEqual(self.TPG, MHTPG)

        from _tpg.trainer import Trainer1
        tpg = self.TPG()
        self.assertIsInstance(tpg.trainer, Trainer1)

    # @unittest.skip('')
    def test_growing(self):
        '''test growing'''
        tpg = self.TPG()
        tpg.setActions(self.actions)
        tpg.setEnv(self.env)
        file = tpg.growing(_dir='test/')
        self.assertIsNotNone(file)
        log_show(f'log/{file}')

if __name__ == '__main__':
    unittest.main()
