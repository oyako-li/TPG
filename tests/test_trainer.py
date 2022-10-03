from datetime import datetime
import unittest

class _TrainerTest(unittest.TestCase):

    def setUp(self) -> None:
        import gym
        from _tpg.trainer import _Trainer
        self.task = "BreakoutNoFrameskip-v4"
        self.env = gym.make(self.task)
        self.Trainer = _Trainer
        self.actions = self.env.action_space.n

    def test_init_(self):
        '''trainer initiation test'''
        trainer = self.Trainer()

        self.assertEqual(len(trainer.teams),0)
        self.assertEqual(len(trainer.rootTeams),0)
        self.assertEqual(len(trainer.learners), 0)
        self.assertEqual(len(trainer.elites), 0)

        self.assertIsNotNone(trainer.teamPopSize)
        self.assertIsNotNone(trainer.rootBasedPop)
        self.assertIsNotNone(trainer.gap)
        self.assertIsNotNone(trainer.inputSize)
        self.assertIsNotNone(trainer.nRegisters)
        self.assertIsNotNone(trainer.initMaxTeamSize)
        self.assertIsNotNone(trainer.initMaxProgSize)
        self.assertIsNotNone(trainer.doElites)
        self.assertIsNotNone(trainer.memMatrix)
        self.assertIsNotNone(trainer.rampancy)
        self.assertIsNotNone(trainer.initMaxActProgSize)
        self.assertIsNotNone(trainer.nActRegisters)
        self.assertIsNotNone(trainer.generation)
        self.assertIsNotNone(trainer.mutateParams)
        self.assertIsNotNone(trainer.actVars)
        self.assertIn("nOperations", trainer.mutateParams)

    def test_set_actions(self):
        ''' test set actions'''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        self.assertNotEqual(len(trainer.teams), 0)

        def allUnique(x):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)
        self.assertTrue(allUnique(trainer.teams))
        self.assertEqual(trainer.ActionObject.actions, range(self.actions))

    # @unittest.skip('')
    def test_get_agents(self):
        ''' test get agents '''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        agents = trainer.getAgents()
        self.assertIsNotNone(agents)
        def allUnique(x):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)

        self.assertTrue(allUnique(agents))

    @unittest.skip("pass impriment")
    def test_get_elite_agent(self):
        ''' test get elite agent '''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        trainer.getEliteAgent()

    # TODO: test save trainer then actions & memories
    @unittest.skip("test complete")
    def test_save_and_load(self):
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        today = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'test/{self.task}/{today}'
        trainer.save(filename)
        self.Trainer.load(filename)

# @unittest.skip('')
class Trainer1Test(_TrainerTest):
    def setUp(self) -> None:
        super().setUp()
        from _tpg.trainer import Trainer1
        self.Trainer = Trainer1

class Trainer1_1Test(Trainer1Test):
    def setUp(self) -> None:
        import gym
        from _tpg.trainer import Trainer1_1
        self.task = "BreakoutNoFrameskip-v4"
        self.env = gym.make(self.task)
        self.Trainer = Trainer1_1
        self.actions = self.env.action_space.n

    def test_set_actions(self):
        ''' test set actions'''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        self.assertNotEqual(len(trainer.teams), 0)

        def allUnique(x):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)
            
        self.assertTrue(allUnique(trainer.teams))
        # print(trainer.ActionObject.actions)

class Trainer2Test(unittest.TestCase):
    def setUp(self) -> None:
        import gym
        self.task = "BreakoutNoFrameskip-v4"
        self.env = gym.make(self.task)
        from _tpg.trainer import Trainer2
        self.Trainer = Trainer2
        self.memories = self.env.observation_space.sample().flatten()
      

    def test_init_(self):
        '''trainer initiation test'''
        trainer = self.Trainer()

        self.assertEqual(len(trainer.teams),0)
        self.assertEqual(len(trainer.rootTeams),0)
        self.assertEqual(len(trainer.learners), 0)
        self.assertEqual(len(trainer.elites), 0)

        self.assertIsNotNone(trainer.teamPopSize)
        self.assertIsNotNone(trainer.rootBasedPop)
        self.assertIsNotNone(trainer.gap)
        self.assertIsNotNone(trainer.inputSize)
        self.assertIsNotNone(trainer.nRegisters)
        self.assertIsNotNone(trainer.initMaxTeamSize)
        self.assertIsNotNone(trainer.initMaxProgSize)
        self.assertIsNotNone(trainer.doElites)
        self.assertIsNotNone(trainer.memMatrix)
        self.assertIsNotNone(trainer.rampancy)
        self.assertIsNotNone(trainer.initMaxActProgSize)
        self.assertIsNotNone(trainer.nMemRegisters)
        self.assertIsNotNone(trainer.generation)
        self.assertIsNotNone(trainer.mutateParams)
        self.assertIsNotNone(trainer.memVars)
        # self.assertIsNotNone(trainer.nOperations)
        self.assertIn("nOperations", trainer.mutateParams)

    def test_set_memories(self):
        ''' test set memories'''
        trainer = self.Trainer()
        trainer.setMemories(self.memories)
        self.assertNotEqual(len(trainer.teams), 0)

        def allUnique(x):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)
        self.assertTrue(allUnique(trainer.teams))

    # @unittest.skip('')
    def test_get_agents(self):
        ''' test get agents '''
        trainer = self.Trainer()
        trainer.setMemories(self.memories)
        agents = trainer.getAgents()
        self.assertIsNotNone(agents)
        def allUnique(x):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)

        self.assertTrue(allUnique(agents))

    @unittest.skip("pass impriment")
    def test_get_elite_agent(self):
        ''' test get elite agent '''
        trainer = self.Trainer()
        trainer.setMemories(self.memories)
        trainer.getEliteAgent()

    # TODO: test save trainer then actions & memories
    # @unittest.skip("test complete")
    def test_save_and_load(self):
        trainer = self.Trainer()
        trainer.setMemories(self.memories)
        today = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'test/{self.task}/{today}'
        trainer.save(filename)
        self.Trainer.load(filename)

if __name__ == '__main__':
    unittest.main()