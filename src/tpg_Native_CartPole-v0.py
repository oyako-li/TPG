import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer1, Trainer
from src.grower import growing

task = 'CartPole-v0'
trainer = Trainer()
filename = growing(trainer, task, _frames=300)
trainer.saveToFile(f'{task}/{filename}')
