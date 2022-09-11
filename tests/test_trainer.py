import unittest

class TrainerTest(unittest.TestCase):

    def setUp(self) -> None:
        import gym
        from _tpg.trainer import _Trainer
        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.trainer = _Trainer()
        self.actions = self.env.action_space.n
        
        return super().setUp()

    @unittest.skip("not implimented yet")
    def test_gym(self):
        pass

    def test_importance(self):
        '''test importance'''
        from _tpg.trainer import _Trainer
        from _tpg.agent import _Agent
        from _tpg.team import _Team
        from _tpg.learner import _Learner
        from _tpg.program import _Program
        from _tpg.action_object import _ActionObject
        _Trainer.importance()
        self.assertEqual(_Trainer.Agent, _Agent)
        self.assertEqual(_Trainer.Team, _Team)
        self.assertEqual(_Trainer.Learner, _Learner)
        self.assertEqual(_Trainer.Program, _Program)
        self.assertEqual(_Trainer.ActionObject, _ActionObject)


if __name__ == '__main__':
    unittest.main()