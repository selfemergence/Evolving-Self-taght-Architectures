# -*- coding: utf-8 -*-

# Custom libraries
from population import Population
from parameter import params

# Third-party libraries
import gym

import numpy as np

from collections import defaultdict

#import cv2

import activation
params['hidden_act_fn'] = activation.sigmoid
params['output_act_fn'] = activation.sigmoid
params['NODE_MUTATE_PROB'] = 0.1 # 0.03
params['CONN_MUTATE_PROB'] = 0.3
params['lr'] = 0.01
      
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

class Simulation:
    """
    Class for doing a simulation using the NEAT Algorithm
    @param env: string, gym environment name
    @param pop_size: int, size of the population
    @param verbosity: int, the level of verbosity [1, 2]. 1 is the lowest level and 2 is the highest level. Optional, defaults to 1
    """
    def __init__(self, env, pop_size, epochs=20, steps=1000, verbosity=1):
        self.env = gym.make(env)
        self._maxSteps = self.env._max_episode_steps
        self.steps = steps
        self.epochs = epochs
        self.pop = Population(pop_size, D, self.env.action_space.n)
        self.verbosity = verbosity
        self.currGen = 1
        
        self.stats = defaultdict(list)

    def run(self, generations, render=False):
        """
        Function for running X number of generations of the simulation. 
        """
        for gen in range(generations):
            print(">Gen ", gen, " Species: ", len(self.pop.species))
            self.stats['species'].append(len(self.pop.species))
            
            for i in range(self.pop.size()):
                agent = self.pop.members[i]
                score = []
                for epoch in range(self.epochs):
                    totalReward = 0
                    observation = self.env.reset()
                    for step in range(self.steps):
                        #convert observation to 80*80 input array
                        observation = prepro(observation)
                        
                        # get output
                        output = agent.network.getOutput(list(observation))
                        #lifetime learning by self-teaching
                        agent.network.self_taught()
                        
                        action = int(np.argmax(output))
                        
                        new_observation, reward, done, info = self.env.step(action)
                        
                        observation = new_observation
                        
                        if (step >= 199):
                            #print ("Failed. Time out")
                            done = True
                            reward -= 20            
                            
                        if done and step < 199:
                            print ("Sucess!")
                            reward += 20
                            
                        totalReward += reward
                        
                        if done:
                            break
                        
                    score.append(totalReward)
                agent.fitness = sum(score)/len(score)
            
            best = max(self.pop.members)
            self.stats['best'].append(best)
            self.stats['best_fitness'].append(best.fitness)
            self.stats['avg_fitness'].append(self.pop.compute_average_fitness())
            print(" Best fitness ", best.fitness, " Conn: ", \
                  len(best.genome.connections), ", Neurons: ", len(best.genome.neurons))
            
            for conn in best.genome.connections.values():
                print("\t", conn)
        
            for neuron in best.genome.neurons.values():
                print("\t", neuron)
                
                                
            #Draw best Network
#            if gen > 0:
#                cv2.imshow("Network", best.network.draw(1000, 800))
#                cv2.waitKey(2)
                
            print("\t innovation number ", params['innov_no'])
            self.stats['innovation'].append(params['innov_no'])
                
            self.pop.evolve()
            
def recordBestBots(bestNeuralNets, env, max_steps):  
    print("\n Recording Best Bots ")
    print("---------------------")
#    env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        for step in range(max_steps):
            env.render()
            outputs = bestNeuralNets[i].network.getOutput(observation)
            action = int(np.argmax(outputs))
            new_observation, reward, done, info = env.step(action)
            totalReward += reward
            observation = new_observation
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % \
              (i+1, bestNeuralNets[i].fitness, totalReward))
        
    env.close()

def main():
    import numpy as np
    np.random.seed(0)
    
    env = 'Pong-v0'
    sim = Simulation(env, 10)
    

    print("================ Starting simulation ================")
    sim.run(1)
    print("================= Ended  simulation =================")
    
    best_fitness = sim.stats['best_fitness']
    avg_fitness = sim.stats['avg_fitness']
    species = sim.stats['species']
    innovation = sim.stats['innovation']
    
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(best_fitness)
    plt.plot(avg_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(['Best', 'Avg'], bbox_to_anchor=(0., 1.02, 1., .102), \
               loc=3, borderaxespad=0., ncol=3, mode="expand")
    plt.savefig('result/fitness.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(species)
    plt.xlabel("Generation")
    plt.ylabel("Species")
    plt.savefig('result/species.png')
    plt.show()
    
    plt.figure(3)
    plt.plot(innovation)
    plt.xlabel("Generation")
    plt.ylabel("Innovation")
    plt.savefig('result/innovation.png')
    plt.show()
    
    
    # Test
    best = max(sim.stats['best'])
    for i_episode in range(20):
        print(">Episode ", i_episode)
        observation = sim.env.reset()
        total_reward = 0
        for step in range(sim._maxSteps):
            sim.env.render()
            output = best.network.getOutput(observation)
            action = int(np.argmax(output))
            new_obs, reward, done, info = sim.env.step(action)
            total_reward += reward
            observation = new_obs
            if done:
                print("\tDone in ", step)
                print("\tTotal reward ", total_reward)
                print("\tEpisode finished after {} timesteps".format(step+1))
                break
    sim.env.close()
    
    bestNeuralNets = sim.stats['best']
    print(bestNeuralNets)
    max_steps = 1000
    recordBestBots(bestNeuralNets, sim.env, max_steps)



if __name__ == '__main__':
    main()