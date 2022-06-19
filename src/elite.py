from _tpg.trainer import Trainer2, loadTrainer
import gym
import matplotlib.pyplot as plt


task = 'CartPole-v0'
env = gym.make(task)

trainer = loadTrainer(f"{task}/2022-06-19_14-31-35")
elietAgent = trainer.getEliteAgent(task)


# import tqdm
def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')


state = env.reset()

for _ in range(500):
    show_state(env, _)
    act = elietAgent.act(state)
    state, reword, isDone, debug = env.step(act)
    if isDone: break