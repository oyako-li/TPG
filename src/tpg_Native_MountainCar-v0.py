import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer1, Trainer
from src.grower import growing

task = 'MountainCar-v0'
trainer = Trainer(teamPopSize=200)
filename = growing(trainer, task, _generations=1000, _frames=500)
trainer.saveToFile(f'{task}/{filename}')

