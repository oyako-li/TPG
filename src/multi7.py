from multiprocessing import Process, Value, Array, Queue, Manager, Pool, Timeout
import gym
import matplotlib.pyplot as plt
import ctypes

def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')

    

def _env(params):
    env = gym.make(params['task'])

    state = env.reset()
    
    while True:
        act = env.action_space.sample()
        # if not actions.empty(): act = actions.get()
        state, reward, isDone, debug = env.step(action=act)
        show_state(env=env)
        if isDone: state = env.reset()


if __name__ == '__main__':
    actions = Queue()
    manager = Manager()
    params = manager.dict()
    params['task'] = 'CartPole-v0'
    src_data = [
        params,
        params
    ]
    try:
        with Timeout(seconds=60):
            with Pool(2) as pool:
                result = pool.starmap(_env, src_data)
                result.get(timeout=5)
    except TimeoutError:
        pass
    