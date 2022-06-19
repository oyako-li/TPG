from _tpg.trainer import Trainer2
import gym
import unittest

class Trainer2Test(unittest.TestCase):
    @unittest.skip("not implimented yet")
    def test_gym(self):
        pass

    def test_emulator(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        trainer = Trainer2()
        actions = env.action_space.n
        trainer.resetActions(actions=actions)
        self.assertEqual(range(actions), trainer.mutateParams['actionCodes'])



if __name__ == '__main__':
    unittest.main()