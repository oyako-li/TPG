from _tpg.trainer import Trainer, Trainer1, loadTrainer
from _tpg.action_object import ActionObject1, ActionObject
from _tpg.memory_object import MemoryObject
import gym
import matplotlib.pyplot as plt


# task = 'MountainCar-v0'
# env = gym.make(task)

# trainer = loadTrainer(f"{task}/2022-06-24_14-57-04")
# elietAgent = trainer.getEliteAgent(task)


# import tqdm
def show_state(env, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')




if __name__ == "__main__":
    import sys
    elietAgent = None
    trainer = None
    task = None

    for arg in sys.argv[1:]:

        if 'task:' in arg: task=arg.split(':')[1]
        if 'model:' in arg:
            modelPath = arg.split(':')[1]
            print(modelPath)
            trainer = loadTrainer(modelPath)
            if isinstance(trainer, Trainer): actions = ActionObject._actions
            elif isinstance(trainer, Trainer1): actions = ActionObject1._actions
    scores = {}
    if task is None:
        tasks=[
            "Acrobot-v1",
            "ALE/Centipede-v5",	
            "ALE/Freeway-v5",		
            "ALE/Riverraid-v5",		
            "Asterix-v4",		
            "Boxing-v0",		
            "CartPole-v0",		
            "CubeCrash-v0",		
            "RoadRunner-v4",
            "WizardOfWor-v4"
        ]

        for task in tasks:

            env = gym.make(task)
            state = env.reset()
            trainer._setUpActions(actions=env.action_space.n)
            elietAgent = trainer.getEliteAgent(task)

            score = 0
            _scores = []
            i=0
            f=0
            print(task)
            while i<10:
                # show_state(env)
                act = env.action_space.sample()
                # act = elietAgent.act(state.flatten())
                # if not act in range(env.action_space.n):
                #     f+=1
                #     if f>500:
                #         print(score)
                #         _scores.append(score)
                #         score=0
                #         state = env.reset()
                #         # if i>10: break
                #         i+=1
                #     continue
                im = elietAgent.image(act, state.flatten())
                state, reward, isDone, debug = env.step(act)
                score += reward
                if isDone:
                    print(score)
                    _scores.append(score)
                    score=0
                    state = env.reset()
                    # if i>10: break
                    i+=1
            scores[task]=max(_scores)
    else:
        env = gym.make(task)
        state = env.reset()
        trainer._setUpActions(actions=env.action_space.n)
        elietAgent = trainer.getEliteAgent(task)

        score = 0
        _scores = []
        i=0
        print(task)
        while True:
                # show_state(env)
                act = elietAgent.act(state.flatten())
                # print(act, ActionObject1._actions, ActionObject._actions)
                if not act in range(env.action_space.n): continue
                state, reward, isDone, debug = env.step(act)
                score += reward
                if isDone:
                    print(score)
                    _scores.append(score)
                    score=0
                    state = env.reset()
                    if i>10: break
                    i+=1
        scores[task]=max(_scores)
        
    print(scores)