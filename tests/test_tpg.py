import unittest
import gym
from _tpg.base_log import *

class TPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import _TPG
        self.TPG = _TPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

    def test_init(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)

    @unittest.skip('next test case')
    def test_episode(self):
        '''test episode'''
        tpg = self.TPG()
        tpg.setActions(self.action)
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
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        score = tpg.generation()
        self.assertIsNotNone(score)

    # @unittest.skip('next test case')
    def test_growing(self):
        '''test growing'''
        tpg = self.TPG()
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        filename = tpg.growing(_dir='test/',_show=True)
        self.assertIsNotNone(filename)
        log_show(f'log/{filename}')

# @unittest.skip('before')
class MHTPGTest(TPGTest):

    def setUp(self) -> None:
        from _tpg.tpg import MHTPG
        self.TPG = MHTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

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
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        file = tpg.growing(_dir='test/')
        self.assertIsNotNone(file)
        log_show(f'log/{file}')

class EmulatorTPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import EmulatorTPG
        self.TPG = EmulatorTPG
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

    def test_init(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)

    @unittest.skip('next test case')
    def test_episode(self):
        '''test episode'''
        tpg = self.TPG()
        tpg.setMemories(self.state)
        tpg.setEnv(self.env)
        tpg.setAgents()
        _scores = {}
        _task = self.env.spec.id
        for _ in range(1):     
            _scores = tpg.episode()
        for i in _scores:               
            _scores[i]/=1
        for agent in tpg.agents: 
            agent.reward(_scores[agent.id],task=_task)
        tpg.trainer.evolve([_task])
        # agents = tpg.getAgents()

    # @unittest.skip('next test case')
    def test_generations(self):
        '''test generation'''
        tpg = self.TPG()
        tpg.setMemories(self.state)
        tpg.setEnv(self.env)
        score = tpg.generation()
        self.assertIsNotNone(score)

    # @unittest.skip('next test case')
    def test_growing(self):
        '''test growing'''
        tpg = self.TPG()
        tpg.setMemories(self.state)
        tpg.setEnv(self.env)
        filename = tpg.growing(_dir='test/')
        self.assertIsNotNone(filename)
        log_show(f'log/{filename}')

class AutomataTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import Automata
        self.Automata = Automata
        self.task = "ALE/Centipede-v5"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n
        self.state = self.env.observation_space.sample().flatten()

    def test_init(self):
        '''test team object creation'''
        from _tpg.trainer import Trainer1, Trainer2
        from _tpg.agent import Agent1, Agent2
        automata = self.Automata()

        self.assertIsNotNone(automata.actor)
        self.assertIsNotNone(automata.emulator)
        self.assertNotEqual(automata.actor.Trainer, automata.emulator.Trainer)
        self.assertIsInstance(automata.actor.trainer, Trainer1)
        self.assertIsInstance(automata.emulator.trainer, Trainer2)
        self.assertEqual(automata.actor.trainer.Agent, Agent1)
        self.assertEqual(automata.emulator.trainer.Agent, Agent2)
        self.assertIsNotNone(automata.actor.trainer.ActionObject.actions)
        self.assertIsNotNone(automata.emulator.Trainer.MemoryObject.memories)

    # @unittest.skip('next test case')
    def test_automata_setup(self):
        '''test setup'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        automata.setAgents()
        self.assertIsNotNone(automata.actor.actions)
        self.assertIsNotNone(automata.actors)
        self.assertIsNotNone(automata.emulator.memories)
        self.assertIsNotNone(automata.emulators)
        # score = automata.generation()
        # self.assertIsNotNone(score)

    @unittest.skip('next test case')
    def test_episode(self):
        '''test episode'''

        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        automata.setAgents()

        _scores = {}
        _task = self.env.spec.id
        for _ in range(1):     
            _scores = automata.episode()
        for i in _scores:               
            _scores[i]/=1
        for agent in automata.agents: 
            agent.reward(_scores[str(agent.team.id)],task=_task)
        automata.evolve([_task])
        # agents = tpg.getAgents()

    # @unittest.skip('next test case')
    def test_growing(self):
        '''test growing'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        filename = automata.growing(_dir='test/')
        self.assertIsNotNone(filename)
        log_show2(f'log/{filename}')

    @unittest.skip('lodad show')
    def test_load(self):
        log_show2('log/test/CartPole-v1/2022-09-29_07-44-55')


if __name__ == '__main__':
    unittest.main()
