import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2


def get_gym_envs():
    for i in gym.envs.registry.all():
        print(i.id)

def get_space_list(space):

    """
    Converts gym `space`, constructed from `types`, to list `space_list`
    """

    # -------------------------------- #

    types = [
        gym.spaces.multi_binary.MultiBinary,
        gym.spaces.discrete.Discrete,
        gym.spaces.multi_discrete.MultiDiscrete,
        gym.spaces.dict.Dict,
        gym.spaces.tuple.Tuple,
        gym.spaces.box.Box,
        gym.spaces.space.Space,
        # "flatdim",
        # "flatten_space",
        # "flatten",
        # "unflatten",
    ]

    if type(space) not in types:
        raise ValueError(f'input space {space} is not constructed from spaces of types:' + '\n' + str(types))

    # -------------------------------- #

    if type(space) is gym.spaces.multi_binary.MultiBinary:
        return [
            np.reshape(np.array(element), space.n)
            for element in itertools.product(
                *[range(2)] * np.prod(space.n)
            )
        ]

    if type(space) is gym.spaces.discrete.Discrete:
        return list(range(space.n))

    if type(space) is gym.spaces.multi_discrete.MultiDiscrete:
        return [
            np.array(element) for element in itertools.product(
                *[range(n) for n in space.nvec]
            )
        ]

    if type(space) is gym.spaces.dict.Dict:

        keys = space.spaces.keys()
        
        values_list = itertools.product(
            *[get_space_list(sub_space) for sub_space in space.spaces.values()]
        )

        return [
            {key: value for key, value in zip(keys, values)}
            for values in values_list
        ]

        return space_list

    if type(space) is gym.spaces.tuple.Tuple:
        return [
            list(element) for element in itertools.product(
                *[get_space_list(sub_space) for sub_space in space.spaces]
            )
        ]
    
    if type(space) is gym.spaces.box.Box:
        return [
            list(element) for element in itertools.product(
                *[get_space_list(sub_space) for sub_space in space.spaces]
            )
        ]

# 空間の出力
def print_spaces(label, space):
   # 空間の出力
   print(label, space)

   # Box/Discreteの場合は最大値と最小値も表示
   if isinstance(space, gym.spaces.Box):
       print('    最小値: ', space.low[0])
       print('    最大値: ', space.high)
       print('    type : ', space.dtype)
   if isinstance(space, gym.spaces.Discrete):
       print('    最小値: ', 0)
       print('    最大値: ', space.n-1)


def show_state(env, step=0, name='', info=''):
    name = env.unwrapped.spec.id
    cv2.imshow(name, env.render(mode='rgb_array'))
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    
# To transform pixel matrix to a single vector.
def get_state(inState):
    # each row is all 1 color
    rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

    # add each with appropriate shifting
    # get RRRRRRRR GGGGGGGG BBBBBBBB
    return np.add(np.left_shift(rgbRows[0], 16), np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))

if __name__ == '__main__':
    # import gym

    task = 'MountainCarContinuous-v0'
    # task = 'Pendulum-v1'
    task = 'CartPole-v0'
    env = gym.make(task) # make the environment
    # print(get_space_list(env.action_space))
    get_gym_envs()

    # print_spaces('行動空間', env.action_space)
