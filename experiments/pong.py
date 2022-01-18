# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(0)

import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

env = gym.make("Pong-v0")
observation = env.reset()

from network import Network
from genome import Genome
from parameter import params

genome = Genome(D, env.action_space.n)
net = Network(genome)

while True:
    env.render()
    total_reward = 0
    observation = prepro(observation)
    
        
    output = net.getOutput(observation)
    #Self-Taught
    net.self_taught()
    
    action = int(np.argmax(output))
    
    new_observation, reward, done, infor = env.step(action)
    
    total_reward += reward
    
    observation = new_observation
    
    genome.mutate()
    net = Network(genome)
    print(len(net.neurons))
    print(len(net.connections))
    
    if done:
        print("REWARD ", total_reward)
        break
    

