import unittest
import gym
import datetime
from _tpg.utils import *

class TPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import _TPG
        self.TPG = _TPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

    @unittest.skip('wrapped initiation')
    def test_init_(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)
        from _tpg.action_object import _ActionObject
        # from _tpg.memory_object import _Memory
        self.assertEqual(tpg.trainer.ActionObject,_ActionObject)
        self.assertIsInstance(tpg.trainer.ActionObject.actions, list)
        # agents = tpg.getAgents()

    @unittest.skip('next test case')
    def test_generations(self):
        '''test generation'''
        tpg = self.TPG()
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        score = tpg.generation()
        self.assertIsNotNone(score)

    @unittest.skip('next test case')
    def test_story(self):
        '''test story'''
        tpg = self.TPG()
        tpg.setEnv(self.env)
        title = tpg.story(_dir='test/', _load=True, _generations=500)
        self.assertIsNotNone(title)
        # log_show(f'{filename}')

    @unittest.skip('prevent logger reset')
    def test_logger(self):
        tpg = self.TPG()
        tpg.setEnv(self.env)
        tpg.setup_logger(__name__,test=True)

    @unittest.skip('test single task')
    def test_muluti_task_learning(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./tasks.txt'):
            with open('./tasks.txt', 'r') as task_file:
                tasks = task_file.readlines()
                print(tasks, type(tasks))
        else:
            tasks = random.choices([
                i.id for i in gym.envs.registry.all()
            ],k=10)
            with open(f'./tasks.txt', 'w') as multi:
                for task in tasks: multi.write(f'{task}\n')
        
        tpg = self.TPG()
        try:
            tpg.multi(tasks, _generations=10, _load=True)
        except Exception as e:
            print(e)
            os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))

    # @unittest.skip('test single task')
    def test_muluti_task_envs(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./tasks.txt'):
            with open('./tasks.txt', 'r') as task_file:
                tasks = task_file.readlines()
                print(tasks, type(tasks))
        else:
            tasks = random.choices([
                i.id for i in gym.envs.registry.all()
            ],k=10)
            with open(f'./tasks.txt', 'w') as multi:
                for task in tasks: multi.write(f'{task}\n')
        
        tpg = self.TPG()
        try:
            tpg.multi(tasks, _generations=1, _load=True)
        except Exception as e:
            print(e)
            os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))
        print(tpg.tasks)

class MHTPGTest(TPGTest):

    def setUp(self) -> None:
        from _tpg.tpg import MHTPG
        self.TPG = MHTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class MHPointTest(MHTPGTest):

    def setUp(self) -> None:
        from _tpg.tpg import MHTPG
        self.TPG = MHTPG
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class ActorTPGPointTest(MHTPGTest):
    def setUp(self) -> None:
        from _tpg.tpg import ActorTPG
        self.TPG = ActorTPG
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

    def test_tpg_setpu(self):
        """tpg setup"""
        tpg = self.TPG()
        tpg.setEnv(self.env)
        tpg.setAgents()

    @unittest.skip('multi_task_test')
    def test_episode(self):
        '''test episode'''
        tpg = self.TPG()
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        _task = self.env.spec.id
        tpg.setAgents(_task)
        for _ in range(1):     
            tpg.episode()
        
        actionSequence = []
        actionReward   = []
        for id in tpg.actionReward:               
            tpg.actionReward[id]/=1.
            actionSequence+=[tpg.actionSequence[id]]
            actionReward+=[tpg.actionReward[id]]
        for agent in tpg.agents: 
            agent.reward(tpg.actionReward[agent.id],task=_task)
        
        tpg.evolve([_task], _actionSequence=actionSequence, _actionReward=actionReward)

class ActorTPGBiasTest(ActorTPGPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import ActorTPG
        self.TPG = ActorTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class ActorPointTest(ActorTPGPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor
        self.TPG = Actor
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n
 
class ActorBiasTest(ActorTPGBiasTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor
        self.TPG = Actor
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class Actor1PointTest(ActorPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor1
        self.TPG = Actor1
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class Actor1BiasTest(ActorBiasTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor1
        self.TPG = Actor1
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class EmulatorTPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import EmulatorTPG
        self.TPG = EmulatorTPG
        self.task = "CartPole-v1"
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

    @unittest.skip('next test case')
    def test_generations(self):
        '''test generation'''
        tpg = self.TPG()
        tpg.setEnv(self.env)
        tpg.setMemories(self.state)
        score = tpg.generation()
        self.assertIsNotNone(score)

    # @unittest.skip('next test case')
    def test_story(self):
        '''test story'''
        tpg = self.TPG()
        tpg.setMemories(self.state)
        tpg.setEnv(self.env)
        filename = tpg.story(_dir='test/')
        self.assertIsNotNone(filename)

class EmulatorEyeTest(EmulatorTPGTest):
    def setUp(self) -> None:
        from _tpg.tpg import EmulatorEye
        self.TPG = EmulatorEye
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

class EmulatorTPG1Test(EmulatorTPGTest):
    def setUp(self) -> None:
        from _tpg.tpg import EmulatorTPG1
        self.TPG = EmulatorTPG1
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

class EmulatorPointTest(EmulatorTPG1Test):
    def setUp(self) -> None:
        from _tpg.tpg import Emulator
        self.TPG = Emulator
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

    def test_init(self):
        '''test initiation'''
        from _tpg.memory_object import Fragment2_1
        from _tpg.trainer import Trainer2_2
        tpg = self.TPG()
        self.assertEqual(tpg.Trainer, Trainer2_2)
        tpg.setEnv(self.env)
        tpg.setMemories(self.state)
        tpg.setAgents()
        self.assertEqual(tpg.Trainer.MemoryObject.memories.Fragment, Fragment2_1)

class EmulatorBiasTest(EmulatorTPG1Test):
    def setUp(self) -> None:
        from _tpg.tpg import Emulator
        self.TPG = Emulator
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

class Emulator1PointTest(EmulatorPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import Emulator1
        self.TPG = Emulator1
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

    def test_init(self):
        '''test initiation'''
        from _tpg.trainer import Trainer2_3
        tpg = self.TPG()
        self.assertEqual(tpg.Trainer, Trainer2_3)
        tpg.setEnv(self.env)
        tpg.setMemories(self.state)
        tpg.setAgents()
        # self.assertEqual(tpg.Trainer.MemoryObject.memories.Fragment, Fragment2_1)

class Emulator1BiasTest(EmulatorBiasTest):
    def setUp(self) -> None:
        from _tpg.tpg import Emulator1
        self.TPG = Emulator1
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.state = self.env.observation_space.sample().flatten()

    def test_init(self):
        '''test initiation'''
        from _tpg.trainer import Trainer2_3
        tpg = self.TPG()
        self.assertEqual(tpg.Trainer, Trainer2_3)
        tpg.setEnv(self.env)
        tpg.setMemories(self.state)
        tpg.setAgents()

class AutomataPointTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import Automata
        self.Automata = Automata
        self.task = "Centipede-v4"
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
    def test_story(self):
        '''test story'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        filename = automata.story(_dir='test/')
        self.assertIsNotNone(filename)
        log_show2(f'log/{filename}')

    @unittest.skip('lodad show')
    def test_load(self):
        log_show2('log/test/CartPole-v1/2022-09-29_07-44-55')

class AutomataBiasTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import Automata
        self.Automata = Automata
        self.task = "CartPole-v1"
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
    def test_story(self):
        '''test story'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        filename = automata.story(_dir='test/')
        self.assertIsNotNone(filename)
        log_show2(f'log/{filename}')

    @unittest.skip('lodad show')
    def test_load(self):
        log_show2('log/test/CartPole-v1/2022-09-29_07-44-55')

class Automata1PointTest(AutomataPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import Automata1
        self.Automata = Automata1
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n
        self.state = self.env.observation_space.sample().flatten()
    

    def test_init(self):
        '''test team object creation'''
        from _tpg.trainer import Trainer1_2, Trainer2_2
        from _tpg.agent import Agent1_1, Agent2_1
        automata = self.Automata()

        self.assertIsNotNone(automata.actor)
        self.assertIsNotNone(automata.emulator)
        self.assertNotEqual(automata.actor.Trainer, automata.emulator.Trainer)
        self.assertIsInstance(automata.actor.trainer, Trainer1_2)
        self.assertIsInstance(automata.emulator.trainer, Trainer2_2)
        self.assertEqual(automata.actor.trainer.Agent, Agent1_1)
        self.assertEqual(automata.emulator.trainer.Agent, Agent2_1)
        self.assertIsNotNone(automata.actor.trainer.ActionObject.actions)
        self.assertIsNotNone(automata.emulator.Trainer.MemoryObject.memories)
    
    def test_story(self):
        '''test story'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        filename = automata.story(_dir='test/')
        self.assertIsNotNone(filename)
        log_show(f'log/{filename}')

class Automata1BiasTest(AutomataBiasTest):
    def setUp(self) -> None:
        from _tpg.tpg import Automata1
        self.Automata = Automata1
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n
        self.state = self.env.observation_space.sample().flatten()

    
    def test_init(self):
        '''test team object creation'''
        from _tpg.trainer import Trainer1_2, Trainer2_2
        from _tpg.agent import Agent1_1, Agent2_1
        automata = self.Automata()

        self.assertIsNotNone(automata.actor)
        self.assertIsNotNone(automata.emulator)
        self.assertNotEqual(automata.actor.Trainer, automata.emulator.Trainer)
        self.assertIsInstance(automata.actor.trainer, Trainer1_2)
        self.assertIsInstance(automata.emulator.trainer, Trainer2_2)
        self.assertEqual(automata.actor.trainer.Agent, Agent1_1)
        self.assertEqual(automata.emulator.trainer.Agent, Agent2_1)
        self.assertIsNotNone(automata.actor.trainer.ActionObject.actions)
        self.assertIsNotNone(automata.emulator.Trainer.MemoryObject.memories)

    def test_story(self):
        '''test story'''
        automata = self.Automata()
        automata.setEnv(self.env)
        automata.setAction(self.action)
        automata.setMemory(self.state)
        filename = automata.story(_dir='test/')
        self.assertIsNotNone(filename)
        log_show(f'log/{filename}')

class LoggingTest(unittest.TestCase):
    def test_logshow(self):
        min, max, ave = log_show('log/test/CartPole-v1/2022-10-30_13-31-37')
        self.assertIsNotNone(min)
        self.assertIsNotNone(max)
        self.assertIsNotNone(ave)


if __name__ == '__main__':
    unittest.main()
