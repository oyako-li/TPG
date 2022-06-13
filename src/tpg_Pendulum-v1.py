import sys
sys.path.insert(0, '.')
from _tpg.trainer import Trainer
from src.grower import growing

task = 'Pendulum-v1'
trainer = Trainer()
filename = growing(trainer, task, _frames=200, _show=True)
trainer.saveToFile(f'{task}/{filename}')

