import time
import numpy as np
import gym
import sys
sys.path.insert(0, '.')

from _tpg.trainer import Trainer
from _tpg.base_log import setup_logger
from tqdm import tqdm
# from src.get_gym_envs import get_space_list, show_state, get_state
from src.grower import generation

task = 'MountainCarC-v0'
logger = setup_logger(__name__, task)
# import to run an agent (always needed)
# from agent import Agent1

# how to render in Jupyter: 
# https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server
# https://www.youtube.com/watch?v=O84KgRt6AJI

env = gym.make(task) # make the environment

tStart = time.time()

# first create an instance of the TpgTrainer
# this creates the whole population and everything
# teamPopSize should realistically be at-least 100
trainer = Trainer(actions=env.action_space.n, teamPopSize=20) 

summaryScores = [] # record score summaries for each gen (min, max, avg)

# 5 generations isn't much (not even close), but some improvements
# should be seen.
for gen in tqdm(range(100)): # generation loop
    trainer, summary = generation(trainer, env, logger, 1800)
    summaryScores.append(summary)
    logger.info(f'generation:{gen}, score:{summary}')

    
    
#clear_output(wait=True)
logger.info(f'Time Taken (Hours): {str((time.time() - tStart)/3600)}')
logger.info(f'Results: Min, Max, Avg, {summaryScores}')