echo $(python -c 'import gym; print([f"{i.id} \n" for i in gym.envs.registry.all()]);') > enable_tasks_next.txt
# echo $(python -c 'import gym; import random; print(random.choices([i.id for i in gym.envs.registry.all()], k=10));') > tasks.txt