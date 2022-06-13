from get_gym_envs import get_space_list, print_spaces

if __name__ == '__main__':
    import gym
    
    task = 'HandManipulatePenTouchSensors-v0'
    # 環境の生成
    env = gym.make(task)

    # 状態空間と行動空間の型の出力
    print('環境ID: ', ENV_ID)
    print_spaces('状態空間: ', env.observation_space)
    print_spaces('行動空間: ', env.action_space)
    
    # get_space_list(task)