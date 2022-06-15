import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer
from src.grower import growing

task = 'ALE/Centipede-v5'
trainer = Trainer(teamPopSize=200)
filename = growing(trainer, task, _generations=1000,_episodes=20, _frames=500)
trainer.saveToFile(f'{task}/{filename}')

