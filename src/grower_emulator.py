import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.insert(0, '.')

import time
import gym
from _tpg.trainer import *
from _tpg.base_log import setup_logger
from tqdm import tqdm
import signal
import time


# import tqdm
def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')

    
# To transform pixel matrix to a single vector.
def getState(inState):
    # each row is all 1 color
    rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

    # add each with appropriate shifting
    # get RRRRRRRR GGGGGGGG BBBBBBBB
    return np.add(np.left_shift(rgbRows[0], 16),
        np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))


# 5 generations isn't much (not even close), but some improvements
# should be seen.
def episode(_agents, _env, _logger=None, _scores={}, _frames:int=100, _show=False):
    
    for agent in _agents: # to multi-proccess
        
        state = _env.reset() # get initial state and prep environment
        score = 0
        _id = str(agent.team.id)
        for _ in range(_frames): # run episodes that last 500 frames
            # while trainer.timeLimit():
            act = agent.act(state)
            # _logger.debug(f'action:{act}')
            # feedback from env
            # state, reward, isDone = emulator.step(act)
            state, reward, isDone, debug = _env.step(act)
            score += reward # accumulate reward in score

            if isDone:  break # end early if losing state
            if _show:   show_state(_env, _)

        if _scores.get(_id) is None : _scores[_id]=0
        _scores[_id] += score # store score

        if _logger is not None: _logger.info(f'{_id},{score}')

    # _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg

    return _scores

# def thinker(_trainer:Trainer2, state):


def generation(_trainer:Trainer2 or Trainer1 or Trainer, _env, _logger=None, _episodes=20, _frames= 100, _show=False):
    _scores = {}
    agents = _trainer.getAgents()
    _task = _env.spec.id
    for _ in range(_episodes):      _scores = episode(agents, _env, _logger=_logger, _scores=_scores, _frames=_frames, _show=_show)
    for i in _scores:                _scores[i]/=_episodes
    for agent in agents:            agent.reward(_scores[str(agent.team.id)],task=_task)
    _trainer.evolve([_task])

    return _scores 


def growing(_trainer:Trainer2 or Trainer1 or Trainer, _task:str, _generations:int=1000, _episodes:int=1, _frames:int=1000, _show=False, _test=False, _load=True):
    logger, filename = setup_logger(__name__, _task, test=_test, load=_load)
    env = gym.make(_task) # make the environment
    action_space = env.action_space
    action = 0
    if isinstance(action_space, gym.spaces.Box):
        action = np.linspace(action_space.low[0], action_space.high[0], dtype=action_space.dtype)
    elif isinstance(action_space, gym.spaces.Discrete):
        action = action_space.n
        # breakpoint(action)
    _trainer.resetActions(actions=action)

    def outHandler(signum, frame):
        _trainer.saveToFile(f'{_task}/{filename}')
        print('exit')
        sys.exit()
    
    signal.signal(signal.SIGINT, outHandler)

    

    summaryScores = []

    tStart = time.time()
    for gen in tqdm(range(_generations)): # generation loop
        scores = generation(_trainer, env, logger, _episodes=_episodes, _frames=_frames, _show=_show)

        score = (min(scores.values()), max(scores.values()), sum(scores.values())/len(scores))

        logger.info(f'generation:{gen}, score:{score}')
        summaryScores.append(score)


        
    #clear_output(wait=True)
    logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
    logger.info(f'Results: Min, Max, Avg, {summaryScores}')
    return filename

if __name__ == '__main__':
    task = sys.argv[1]
    show = False
    test = False
    load = False

    for arg in sys.argv[2:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
    trainer = Trainer2(teamPopSize=10)
    _filename = growing(trainer, task, _show=show, _test=test, _load=load)
    trainer.saveToFile(f'{task}/{_filename}')

