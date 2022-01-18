import gym

env = gym.make('BipedalWalker-v3')
env.seed(2021)
env.action_space.seed(2021)


observation = env.reset()
action = env.action_space.sample()

print(observation[:2], action[:2])