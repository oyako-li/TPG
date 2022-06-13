import multiprocessing as mp
import time
import gym
from trainer import Trainer
from agent import Agent
from tqdm import tqdm

import numpy as np
import gym
import matplotlib.pyplot as plt
import random
# import to do training
from trainer import Trainer
# import to run an agent (always needed)
from agent import Agent

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
    return np.add(np.left_shift(rgbRows[0], 16), np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))

def runAgent(args:list[Agent, str, list, int, int]):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numEpisodes = args[3] # number of times to repeat game
    numFrames = args[4] 
    
    # skip if task already done by agent
    if agent.taskDone(envName):
        print('Agent #' + str(agent.agentNum) + ' can skip.')
        scoreList.append((agent.team.id, agent.team.outcomes))
        return
    
    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less
    
    scoreTotal = 0 # score accumulates over all episodes
    for ep in range(numEpisodes): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = 0
        if numEpisodes > 1:
            numRandFrames = random.randint(0,30)
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                env.step(env.action_space.sample())
                continue

            act = agent.act(getState(np.array(state, dtype=np.int32)))

            # feedback from env
            state, reward, isDone, debug = env.step(act)
            scoreEp += reward # accumulate reward in score
            if isDone:
                break # end early if losing state
                
        print('Agent #' + str(agent.agentNum) + 
              ' | Ep #' + str(ep) + ' | Score: ' + str(scoreEp))
        scoreTotal += scoreEp
       
    scoreTotal /= numEpisodes
    env.close()
    agent.reward(scoreTotal, envName)
    scoreList.append((agent.team.id, agent.team.outcomes))

if __name__ == '__main__':
    envName = 'Boxing-v0'
    # get num actions
    env = gym.make(envName)
    acts = env.action_space.n
    del env

    trainer = Trainer(actions=acts, teamPopSize=360)

    processes = 10
    man = mp.Manager()
    pool = mp.Pool(processes=processes, maxtasksperchild=1)
        
    allScores = [] # track all scores each generation

    tStart = time.time()
    for gen in tqdm(range(100)): # do 100 generations of training
        scoreList = man.list()
        
        # get agents, noRef to not hold reference to trainer in each one
        # don't need reference to trainer in multiprocessing
        agents = trainer.getAgents() # swap out agents only at start of generation

        process_seed = [[agent, envName, scoreList, 1, 18000] for agent in agents]

        # run the agents
        pool.map(runAgent, process_seed)
        
        # apply scores, must do this when multiprocessing
        # because agents can't refer to trainer
        teams = trainer.applyScores(scoreList)
        # important to remember to set tasks right, unless not using task names
        # task name set in runAgent()
        trainer.evolve(tasks=[envName]) # go into next gen
        
        # an easier way to track stats than the above example
        scoreStats = trainer.fitnessStats
        allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
        
        # clear_output()
        print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
        print('Gen: ' + str(gen))
        print('Results so far: ' + str(allScores))
        
    # clear_output()
    print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
    print('Results:\nMin, Max, Avg')
    for score in allScores:
        print(score[0],score[1],score[2])