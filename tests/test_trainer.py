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
        # self.assertIsNotNone(trainer.traversal)
        self.assertIsNotNone(trainer.initMaxActProgSize)
        self.assertIsNotNone(trainer.nActRegisters)
        self.assertIsNotNone(trainer.generation)
        self.assertIsNotNone(trainer.mutateParams)
        self.assertIsNotNone(trainer.actVars)
        self.assertIsNotNone(trainer.nOperations)
        # self.assertIsNotNone(trainer.operations)
        self.assertIn("nOperations", trainer.mutateParams)

    def test_set_actions(self):
        ''' test set actions'''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        self.assertNotEqual(len(trainer.teams),0)
        self.assertNotEqual(len(trainer.rootTeams),0)
        self.assertNotEqual(len(trainer.learners),0)

    def test_get_agents(self):
        ''' test get agents '''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        self.assertIsNotNone(trainer.getAgents())

    @unittest.skip("pass impriment")
    def test_get_elite_agent(self):
        ''' test get elite agent '''
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        trainer.getAgents()

    # TODO: test save trainer then actions & memories
    @unittest.skip("test complete")
    def test_save_and_load(self):
        trainer = self.Trainer()
        trainer.setActions(self.actions)
        today = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'test/{self.task}/{today}'
        trainer.save(filename)
        self.Trainer.load(filename)

class Trainer1Test(_TrainerTest):
    def setUp(self) -> None:
        super().setUp()
        from _tpg.trainer import Trainer1
        self.Trainer = Trainer1

if __name__ == '__main__':
    unittest.main()