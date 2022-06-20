import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer1
from src.grower_hierarchical import growing

task = 'RoadRunner-v4'
trainer = Trainer1(teamPopSize=100)
filename = growing(trainer, task, _generations=1000,_episodes=1, _frames=500, _load=False)
trainer.saveToFile(f'{task}/{filename}')

