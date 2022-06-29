# from src.grower import growing
from _tpg.trainer import loadTrainer
from _tpg.tpg import NativeTPG, MemoryAndHierarchicalTPG, EmulatorTPG
import sys

def breakpoint(_print):
    print(_print)
    sys.exit()

if __name__ == '__main__':
    task = 'CartPole-v0'
    show = False
    test = False
    load = False
    trainer = None
    teamPopSize=100
    generations=1000
    episodes=1
    frames=500



    for arg in sys.argv[1:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
        if 'teamPopSize:' in arg: teamPopSize=int(arg.split(':')[1])
        if 'generatioins:' in arg: generations=int(arg.split(':')[1])
        if 'episodes:' in arg: episodes=int(arg.split(':')[1])
        if 'frames:' in arg: frames=int(arg.split(':')[1])
        # if 'CubeCrash_pattern:' in arg: pattern = arg.split(':')[1]


    for arg in sys.argv[1:]:
        if arg=='native':
            tpg = NativeTPG(teamPopSize=teamPopSize)
        elif arg=='hierarchy':
            tpg = MemoryAndHierarchicalTPG(teamPopSize=teamPopSize)
        elif arg=='emulator':
            tpg = EmulatorTPG(teamPopSize=teamPopSize)
        if 'model:' in arg:
            modelPath = arg.split(':')[1]
            task = '/'.join(modelPath.split('/')[:-1])
            # breakpoint(task)
            if task == "Acrobot-v1":
                tasks = [
                    "ALE/Centipede-v5",
                    "ALE/Freeway-v5",
                    "ALE/Riverraid-v5",
                    "Asterix-v4",
                    "Boxing-v0",
                    "CartPole-v0"
                ]
            elif task=="CartPole-v0":
                tasks = [
                    "CubeCrash-v0",
                    "WizardOfWor-v4"
                    "ALE/Freeway-v5",
                    "ALE/Riverraid-v5",
                    "Acrobot-v1",
                    "Asterix-v4",
                    "CartPole-v0",
                    "ALE/Centipede-v5",
                    "RoadRunner-v4"
                ]
            if task == "ALE/Freeway-v5":
                tasks = [
                    "ALE/Centipede-v5"
                ]
            elif task == "ALE/Centipede-v5":
                tasks = [
                    "CubeCrash-v0",
                    "RoadRunner-v4",
                    "WizardOfWor-v4",
                    "Acrobot-v1",
                    "RoadRunner-v4",
                    "CartPole-v0"
                ]

            if task=="Boxing-v0":
                tasks = [
                    "CubeCrash-v0",
                    "Asterix-v4",
                    "Boxing-v0",
                    "ALE/Riverraid-v5",
                    "WizardOfWor-v4",
                    "Acrobot-v1",
                    "RoadRunner-v4",
                    "CartPole-v0",
                ]
            # filename = modelPath.split('/')[-1]
            trainer = loadTrainer(modelPath)

    if not tpg: raise Exception('TPG type is not defined')

    for task in tasks:
        filename = tpg.start(_task=task, _show=show, _test=test, _load=load, _trainer=trainer, _generations=generations, _episodes=episodes, _frames=frames)
        trainer = loadTrainer(f'{task}/{filename}')