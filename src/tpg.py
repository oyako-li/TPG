import sys
sys.path.insert(0, '.')
from src.grower import growing

if __name__=='__main__':
    task = 'ALE/JourneyEscape-v5'
    trainer = growing(task)