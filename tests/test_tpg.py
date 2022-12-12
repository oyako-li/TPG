import unittest
import gym
import datetime
import random
import sys
from _tpg.utils import *

class TPGTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.tpg import _TPG
        self.TPG = _TPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

    # @unittest.skip('wrapped initiation')
    def test_init_(self):
        '''test team object creation'''
        tpg = self.TPG()
        self.assertIsNotNone(tpg.Trainer)
        from _tpg.action_object import _ActionObject
        # from _tpg.memory_object import _Memory
        self.assertEqual(tpg.trainer.ActionObject,_ActionObject)
        self.assertIsInstance(tpg.trainer.ActionObject.actions, list)

    # @unittest.skip('next test case')
    def test_generations(self):
        '''test generation'''
        tpg = self.TPG()
        tpg.setActions(self.action)
        tpg.setEnv(self.env)
        score = tpg.generation()
        self.assertIsNotNone(score)

    # @unittest.skip('prevent logger reset')
    def test_logger(self):
        tpg = self.TPG()
        tpg.setEnv(self.env)
        tpg.setup_logger(__name__,test=True)
    
    def test_save_load(self):

        tpg = self.TPG()
        tpg.setEnv(self.env)
        title = tpg.story(_dir='test/', _load=True, _generations=1)
        tpg.load_story(title)
        new_title = tpg.story(_dir='test/', _load=True, _generations=1)
        print(title, new_title)


    # @unittest.skip('next test case')
    def test_single(self):
        '''test story'''
        tpg = self.TPG()
        times = 1
        generations=100
        if args := sys.argv[2:]:
            for arg in args:
                if 'task:' in arg:
                    self.env = gym.make(arg.split(':')[1])
                if 'times:' in arg:
                    times = int(arg.split(':')[1])
                if 'show' == arg:
                    tpg.show = True
                if 'generations:' in arg:
                    generations = int(arg.split(':')[1])
        archive = set()
        for _ in range(times):
            tpg.setEnv(self.env)
            title = tpg.story(_dir=f'{tpg.task}'.replace('/','-')+'/', _load=True, _generations=generations)
            # tpg.restert()
            tpg = self.TPG()
            tpg.unset_logger()
            archive.add(title)
        self.assertEqual(set(tpg.archive))

    def test_single_each(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./.tasks'):
            with open('./.tasks', 'r') as task_file:
                tasks = task_file.read().splitlines()
                random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))
                random.shuffle(tasks)
                print(tasks, type(tasks))
        else:
            raise Exception('tasksDoesntExist')
        
        try:
            for _ in range(10):
                for task in tasks:
                    tpg = self.TPG()
                    tpg.unset_logger()
                    # tpg.setEnv(gym.make(task))
                    tpg.story(_task=task, _generations=100, _load=True)
        except Exception as e:
            print(e)
            # os.remove('./tasks.txt')
        # self.assertTrue(tpg.tasks!=set(tasks))
    
    # @unittest.skip('test single task')
    def test_multi(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./.tasks'):
            with open('./.tasks', 'r') as task_file:
                tasks = task_file.read().splitlines()
                print(tasks, type(tasks))
                random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))
        else:
            raise Exception('tasksDoesntExist')
        
        try:
            for _ in range(10):
                tpg = self.TPG()
                random.shuffle(tasks)
                tpg.multi(tasks, _generations=100, _load=True)
                tpg.unset_logger()

        except Exception as e:
            print(e)
            # os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))

    # @unittest.skip('test single task')
    def test_multi_random(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./.tasks'):
            with open('./.tasks', 'r') as task_file:
                tasks = task_file.read().splitlines()
                random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))

                random.shuffle(tasks)
                print(tasks, type(tasks))
        else:
            raise Exception('tasksDoesntExist')
        
        tpg = self.TPG()
        try:
            for _ in range(10):
                tpg.multi(tasks, _generations=10, _load=True)
                tpg.unset_logger()
                random.shuffle(tasks)

        except Exception as e:
            print(e)
            # os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))

    # @unittest.skip('test single task')
    def test_multi_chaos(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./.tasks'):
            with open('./.tasks', 'r') as task_file:
                tasks = task_file.read().splitlines()
                random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))

                print(tasks, type(tasks))
        else:
            raise Exception('tasksDoesntExist')
        times = 10
        if args := sys.argv[2:]:
            for arg in args:
                if 'times:' in arg:
                    times = int(arg.split(':')[1])
        tpg = self.TPG()
        try:
            for _ in range(times):
                tpg.chaos_story(_tasks=tasks, _generations=100, _load=True)
                tpg.unset_logger()
                tpg = self.TPG()
        except Exception as e:
            print(e)
            # os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))

    def test_multi_elite(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        title='log/'
        _title='log/'
        tpg = self.TPG()
        if args := sys.argv[2:]:
            for arg in args:
                if 'title:' in arg:
                    title = arg.split(':')[1]

        if os.path.exists('./.tasks'):
            with open('./.tasks', 'r') as task_file:
                tasks = task_file.read().splitlines()
                # random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))
                # random.shuffle(tasks)
                # print(tasks, type(tasks))
        else:
            raise Exception('tasksDoesntExist')
    
        if os.path.exists(f'{title}.pickle'):
            tpg.load_story(title)
        else:
            raise Exception('titleDoesntExist')
        
        archive = []
        try:
            for task in tasks:
                _title = tpg.success_story(_task=task)
                archive.append(_title)
        except Exception as e:
            print(e)
            # os.remove('./tasks.txt')
        print(archive)

    # @unittest.skip('test single task')
    def test_multi_envs(self):
        """ マルチタスク学習に対応できるように、改良。
        """
        tasks=[]
        if os.path.exists('./tasks.txt'):
            with open('./tasks.txt', 'r') as task_file:
                tasks = task_file.read().splitlines()
                random.seed(datetime.now().strftime('%Y%m%d%H%M%S'))
                random.shuffle(tasks)
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
            with open(f'./.tasks', 'w') as multi:
                for task in tasks: multi.write(f'{task}\n')
        except Exception as e:
            print(e)
            os.remove('./tasks.txt')
        self.assertEqual(tpg.tasks, set(tasks))
        print(tpg.tasks)

    def test_argv(self):
        print(sys.argv[2:])

    def test_log(self):
        title='log/test/2022-11-30/20-24-11'
        task = 'CartPole-v1'
        if args := sys.argv[2:]:
            for arg in args:
                if 'task:' in arg:
                    task = arg.split(':')[1]
                elif 'title:' in arg:
                    title = arg.split(':')[1]

        log_show(_title=title,_task=task)

class MTPGTest(TPGTest):
    def setUp(self) -> None:
        from _tpg.tpg import MTPG
        self.TPG = MTPG
        self.task = "CartPole-v1"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class MHTPGTest(MTPGTest):

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

class Actor2PointTest(ActorPointTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor2
        self.TPG = Actor2
        self.task = "Centipede-v4"
        self.env = gym.make(self.task)
        self.action = self.env.action_space.n

class Actor2BiasTest(ActorBiasTest):
    def setUp(self) -> None:
        from _tpg.tpg import Actor2
        self.TPG = Actor2
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
