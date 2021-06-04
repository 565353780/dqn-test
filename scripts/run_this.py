from maze_env import Maze
from RL_brain import DeepQNetwork

import os
from RobotScanning2DEnvironment import RobotScanning2DEnvironment

os.environ['CUDA_VISIBLE_DEVICES']='0'


def run_maze(checkpoint_idx):
    step = 0
    for episode in range(checkpoint_idx + 1, 10000000):
        # initial observation
        observation = env.reset()

        total_reward = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            total_reward += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print('episode : ', episode, '  //  ', end='')
                print('reward : ', total_reward)
                with open('data.txt', 'a+') as f:
                    f.write('{' + str(episode) + ',' + str(total_reward) + '},\n')
                break
            step += 1

        if episode % 100 == 0:
            RL.save(episode)

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    # env = Maze()
    env = RobotScanning2DEnvironment('./32x32.png')
    checkpoint_idx = 0
    if checkpoint_idx == 0:
        file_list = os.listdir(os.getcwd() + '/checkpoints/')
        for file_name in file_list:
            if 'index' in file_name:
                if int(file_name.split('.')[0].split('-')[1]) > checkpoint_idx:
                    checkpoint_idx = int(file_name.split('.')[0].split('-')[1])

    RL = DeepQNetwork(8, 3072,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True,
                      checkpoint_idx=checkpoint_idx
                      )
    # env.after(100, run_maze)
    run_maze(checkpoint_idx)
    # env.mainloop()
    RL.plot_cost()
