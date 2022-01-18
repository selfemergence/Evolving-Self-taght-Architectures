# relative import from another directory
import os
import sys
p = os.path.abspath('../')
sys.path.insert(1, p)

# Custom libraries
from population import Population
from parameter import params

from torch.multiprocessing import Pool
from functools import partial

# Third-party libraries
import gym
import copy
import time

import numpy as np

from collections import defaultdict

import cv2

import activation
params['hidden_act_fn'] = activation.relu
params['output_act_fn'] = activation.sigmoid
params['NODE_MUTATE_PROB'] = 0.1 # 0.03, 0.8
params['CONN_MUTATE_PROB'] = 0.5


def run_agent(agent, env, epochs=100, steps=1000, render_test=False):
    
    score = []
    #agent_env = copy.deepcopy(env)
    for epoch in range(epochs):
        if render_test:
            print("***Testing Epoch ", epoch)
        total_reward = 0
        observation = env.reset()
        done = False
        st = 0
        while not done:
            if render_test:
                env.render()
                time.sleep(0.005)
            st += 1
            output = agent.network.getOutput(list(observation))
            # lifetime learning by self-teaching
#            if not render_test:
#                agent.network.self_taught()
            
            action = int(np.argmax(output))
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            if done or st == env._max_episode_steps:
                break
        score.append(total_reward)
        average_reward = np.average(score)
        if render_test:
            print(f'\tReward at epoch {epoch} is {total_reward}')
            print(f'\tAverage Reward at after {epoch} epoch(s) is {average_reward}')
    
    env.close()    
    
    return average_reward
            

class Simulation:
    """
    Class for doing a simulation using the NEAT Algorithm
    @param env: string, gym environment name
    @param pop_size: int, size of the population
    @param verbosity: int, the level of verbosity [1, 2]. 1 is the lowest level and 2 is the highest level. Optional, defaults to 1
    """
    def __init__(self, env, pop_size, epochs=10, steps=1000, verbosity=1):
        self.env = env
        self.env.seed(2021)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self._maxSteps = self.env._max_episode_steps
        self.steps = steps
        self.epochs = epochs
        self.pop = Population(pop_size, self.state_size, 
                              self.action_size)
        self.verbosity = verbosity
        self.currGen = 1
        
        self.pool = Pool(processes=3)
        
        self.stats = defaultdict(list)

    def run(self, generations, render=False):
        """
        Function for running X number of generations of the simulation. 
        """
        for gen in range(generations):
            print(">Gen ", gen, " Species: ", len(self.pop.species))
            self.stats['species'].append(len(self.pop.species))
            
            #using multiprocessing pool
            fitness_function = partial(run_agent, env=copy.deepcopy(self.env), 
                                       epochs=self.epochs, steps=self._maxSteps)
            fitnesses = self.pool.map(fitness_function, [self.pop.members[i] for i in range(self.pop.size())])
            for i in range(self.pop.size()):
                self.pop.members[i].fitness = fitnesses[i]
            
            best = max(self.pop.members)
            self.stats['best'].append(best)
            self.stats['best_fitness'].append(best.fitness)
            self.stats['avg_fitness'].append(self.pop.compute_average_fitness())
            print(" Best fitness ", best.fitness, " | average fitness ", round(self.stats['avg_fitness'][-1], 2))
            print(" Conn: ", len(best.genome.connections), ", Neurons: ", len(best.genome.neurons))
            self.stats['innovation'].append(params['innov_no'])
            
            # evolve population
            self.pop.evolve()
            
    
        
        return best

def main():
    
    env_name = 'maze-sample-10x10-v0'
    env = gym.make(env_name)
    env.seed(2021)
    np.random.seed(2021)
    
    # evolutionary parameters
    generations = 51
    pop_size = 200
    epochs = 1
    steps = env._max_episode_steps
    params['elitism'] = pop_size//20
    
    sim = Simulation(env, pop_size=pop_size, epochs=epochs, steps=steps)
    

    print("================ Starting simulation ================")
    solution = sim.run(generations)
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
    plt.savefig('result/' + env_name + '-fitness.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(species)
    plt.xlabel("Generation")
    plt.ylabel("Species")
    plt.savefig('result/' + env_name +'-species.png')
    plt.show()
    
    plt.figure(3)
    plt.plot(innovation)
    plt.xlabel("Generation")
    plt.ylabel("Innovation")
    plt.savefig('result/' + env_name +'-innovation.png')
    plt.show() 
    
    # Test
    best = max(sim.stats['best'])
    
    # save solution
    print("***Saving the best solution")
    import pickle
    # save best weights for future uses
    with open('result/' + env_name + '-bests-st.plt', 'wb') as f:
        pickle.dump(sim.stats['best'], f)

    print("***Running the best solution")
    
    score = []
    test_epochs = 100
    for i_episode in range(test_epochs):
        print(">Episode ", i_episode)
        observation = sim.env.reset()
        total_reward = 0
        for step in range(steps):
            sim.env.render()
            output = best.network.getOutput(observation)
            #lifetime learning by self-teaching
            #best.network.self_taught()
            
            action = int(np.argmax(output))
            new_obs, reward, done, info = sim.env.step(action)
            total_reward += reward
            observation = new_obs
            if done:
                print("\tDone in ", step)
                print("\tTotal reward ", total_reward)
                print("\tEpisode finished after {} timesteps".format(step+1))
                break
        score.append(total_reward)
    print(f'Average reward after {test_epochs} is {np.average(score)}')
    sim.env.close()
    env.close()
    
    #Draw best Network
    img = solution.network.draw(1000, 800)
    outpath = 'result/' + env_name + '-network.png'
    ##save the image
    #img = network.draw(width, height)
    cv2.imwrite(outpath, img)
    cv2.waitKey(0)
        
    

#    # load solution
#    import pickle
#    # save best weights for future uses
#    file = open('result/' + env_name + 'bests-st.plt','rb')
#    bests = pickle.load(file)
#    file.close()
#    
#    best_agent = max(bests)
#    print("***Expected Reward ", best_agent.fitness)
#    print("***Running Best ")
#    average_reward = run_agent(best_agent, sim.env, render_test=True)
#    print("*** Average reward after 100 epochs is ", average_reward)


if __name__ == '__main__':
    main()