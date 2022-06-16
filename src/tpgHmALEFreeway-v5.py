import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer1
from src.grower_hierarchical import growing

task = 'ALE/Freeway-v5'
trainer = Trainer1(teamPopSize=200)
filename = growing(trainer, task, _generations=1000,_episodes=20, _frames=200, _load=False)
trainer.saveToFile(f'{task}/{filename}')

