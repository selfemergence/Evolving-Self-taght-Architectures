import gym
import gym_maze
import copy

#env = gym.make("maze-random-10x10-plus-v0")
#env = gym.make("maze-sample-100x100-v0")
#env = gym.make("maze-random-30x30-plus-v0")
env_name= "maze-sample-10x10-v0"
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
max_steps = env._max_episode_steps
threshold = env.spec.reward_threshold
print(state_size, action_size, max_steps, threshold)

#print(dir(env.action_space))

#env1 = copy.copy(env)
#env2 = copy.copy(env)

for i in range(10):
    print(f"*** RUNNING ENVIRONMENT {i+1}")
    copy_env = copy.copy(env)
    observation = copy_env.reset()
    st = 0
    while True:
        st += 1
        copy_env.render()
        action = copy_env.action_space.sample()
        #print(observation, action)
        observation, reward, done, info = copy_env.step(action)
        
        if done or (st == copy_env._max_episode_steps - 1):
            copy_env.close()
            break

#done = False
#observation = env.reset()
#while True:
#    env.render()
#    action = env.action_space.sample()
#    print(observation, action)
#    observation, reward, done, info = env.step(action)
    