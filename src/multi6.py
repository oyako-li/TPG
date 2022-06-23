from multiprocessing import Process, Value, Array, Queue, Manager, Pool, Pipe
import gym
import matplotlib.pyplot as plt
import time

def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')

    

def _env(params, _actions, _states):
    env = gym.make(params['task'])

    state = env.reset()
    while True:
        act = env.action_space.sample()
        if not _actions.empty(): 
            act = _actions.get()
            print(act)
        state, reward, isDone, debug = env.step(action=act)
        # connection.send(state)
        _states.put(state)
        show_state(env=env)
        if isDone: state = env.reset()

class Result():
    def __init__(self, actions) -> None:
        self.actions = actions

def update_result(val):
    print('think')
    return val


def thinking(_states):
    time.sleep(0.1)

    if not _states.empty(): 
        state = _states.get(timeout=0.2)
        return state

    return 0

def _actor(params, actions, states):
    # results = Result(actions=actions)
    with Pool(2) as pool:
        results = pool.apply_async(thinking, states, callback=update_result)
        while True:
            try:
                result = results.get(timeout=10)
                actions.put(result)
            except TimeoutError:
                pass
        # results.get(timeout=0.2)



if __name__ == '__main__':
    actions = Queue()
    states = Queue()
    envpipe, actorpipe = Pipe()
    manager = Manager()
    params = manager.dict()
    params['task'] = 'CartPole-v1'
    # params['state']= 0

    env = Process(target=_env, args=(params, actions, states))
    actor = Process(target=_actor, args=(params, actions, states))
    env.start()
    actor.start()
    env.join(timeout=10)
    actor.join(timeout=10)
    envpipe.close()
    actorpipe.close()
    env.terminate()
    actor.terminate()
