from _tpg.trainer import Trainer2, loadTrainer
import gym
import matplotlib.pyplot as plt


task = 'MountainCar-v0'
env = gym.make(task)

trainer = loadTrainer(f"{task}/2022-06-24_14-57-04")
elietAgent = trainer.getEliteAgent(task)


# import tqdm
def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')


state = env.reset()

while True:
    show_state(env)
    act = elietAgent.act(state)
    state, reword, isDone, debug = env.step(act)
    if isDone: state = env.reset()