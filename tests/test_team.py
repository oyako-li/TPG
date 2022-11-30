import unittest
import uuid
from _tpg.team import *

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

class Team1_2_1Test(TeamTest):
    def setUp(self) -> None:
        from _tpg.team import Team1_2_1
        self.Team = Team1_2_1

    def test_init(self):
        '''test team object creation'''
        team = self.Team()
        self.assertTrue(isinstance(team._id, uuid.UUID))

    def test_clone(self):
        team = self.Team()
        self.assertTrue(
            team._id != team.clone._id
        )

if __name__ == '__main__':
    unittest.main()
