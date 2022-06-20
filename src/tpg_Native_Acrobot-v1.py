import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer
from src.grower import growing

task = 'Acrobot-v1'
trainer = Trainer(teamPopSize=100)
filename = growing(trainer, task, _generations=1000,_episodes=1, _frames=500, _load=False)
trainer.saveToFile(f'{task}/{filename}')

