import numpy as np
import matplotlib.pyplot as plt
# from IPython import display
import sys
sys.path.insert(0, '.')

import time
import gym
from _tpg.trainer import Trainer
from _tpg.base_log import setup_logger
from tqdm import tqdm

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
def generation(_trainer, _env, _logger=None, _1_episode_frames:int=100):
    curScores = [] # new list per gen
    
    agents = _trainer.getAgents()
    
    for agent in agents: # to multi-proccess
        
        state = _env.reset() # get initial state and prep environment
        score = 0
        for _ in range(_1_episode_frames): # run episodes that last 500 frames
            # show_state(
            #     _env, _, 'Assault',
            #            ' Score: ' + str(score)
            # )
            act = agent.act(state) 

            # feedback from env
            state, reward, isDone, debug = _env.step(act)
            score += reward # accumulate reward in score

            if isDone: break # end early if losing state

        agent.reward(score) # must reward agent (if didn't already score)
            
        curScores.append(score) # store score
        if _logger is not None: _logger.info(f'{agent.team.id},{score}')

    _summaryScores = (min(curScores), max(curScores), sum(curScores)/len(curScores)) # min, max, avg
    _trainer.evolve()

    return _trainer, _summaryScores

def growing(_task, _generations=2000):
    logger = setup_logger(__name__, _task)


    env = gym.make(_task) # make the environment

    tStart = time.time()

    # first create an instance of the TpgTrainer
    # this creates the whole population and everything
    # teamPopSize should realistically be at-least 100
    trainer = Trainer(actions=env.action_space) 

    summaryScores = [] # record score summaries for each gen (min, max, avg)

    for gen in tqdm(range(_generations)): # generation loop
        trainer, summary = generation(trainer, env, logger, 100)
        summaryScores.append(summary)
        logger.info(f'generation:{gen}, score:{summary}')

        
        
    #clear_output(wait=True)
    logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
    logger.info(f'Results: Min, Max, Avg, {summaryScores}')
    return trainer

if __name__ == '__main__':

    task = 'MountainCarContinuous-v0'
    trainer = growing(task)
    trainer.saveToFile(task)

