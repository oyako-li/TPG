
import gym
import ppaquette_gym_super_mario
import numpy as np

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

for episode in range(5):
  observation = env.reset()                             # 環境の初期化
  for _ in range(100):
    action = np.random.randint(0, 1+1, 6)               # 行動の決定
    observation, reward, done, info = env.step(action)  # 行動による次の状態の決定
    print("=" * 10)
    print("action=",action)
    print("observation=",observation)
    print("reward=",reward)
    print("done=",done)
    print("info=",info)

env.close()                                             # GUI環境の終了
