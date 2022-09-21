import unittest

class TeamTest(unittest.TestCase):
    def setUp(self) -> None:
        from _tpg.team import _Team
        self.Team = _Team

    def test_init(self):
        '''test team object creation'''
        team = self.Team()

        self.assertIsNotNone(self.Team.Learner)
        self.assertIsNotNone(team.learners)
        self.assertIsNotNone(team.outcomes)
        self.assertIsNone(team.fitness)
        self.assertIsNotNone(team.inLearners)
        self.assertIsNotNone(team.id)
        self.assertIsNotNone(team.genCreate)

    def test_act(self):
        '''test act'''


if __name__ == '__main__':
    unittest.main()
