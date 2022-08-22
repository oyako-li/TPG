from _tpg.trainer import Trainer2
from _tpg.tpg import EmulatorTPG
import gym
import unittest

class Trainer2Test(unittest.TestCase):

    def test_emulator(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        emulator = Trainer2(teamPopSize=10)
        state = env.reset()
        emulator.resetMemories(state=state.flatten())
        agent = emulator.getAgents()[0]
        # self.assertEqual(range(actions), trainer.mutateParams['actionCodes'])
        for _ in range(100):
            act = env.action_space.sample()
            imageCode, reward =  agent.image(act, state.flatten())
            print(state, reward)
            state, reward, isdone, debug = env.step(act)



if __name__ == '__main__':
    unittest.main()