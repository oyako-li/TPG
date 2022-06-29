# from src.grower import growing
from _tpg.trainer import loadTrainer
import sys


if __name__ == '__main__':

    # task = 'CartPole-v0'
    task = sys.argv[1]
    show = False
    test = False
    load = False
    frame=500
    model = sys.argv[2]

    for arg in sys.argv[3:]:
        if arg=='show': show = True
        if arg=='test': test=True
        if arg=='load': load=True
        if arg=='native':
            from src.grower import growing
        if arg=='hierarchy':
            from src.grower_hierarchical import growing
        if arg=='emulator':
            from src.grower_emulator import growing
        if isinstance(arg, int): frame=arg
    trainer = loadTrainer(model)
    _filename = growing(trainer, task, _episodes=1, _frames=frame, _show=show, _test=test, _load=load)
    trainer.saveToFile(f'{task}/{_filename}')