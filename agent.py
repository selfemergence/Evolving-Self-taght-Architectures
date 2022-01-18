# -*- coding: utf-8 -*-
#import sys

from genome import Genome
from network import Network
import numpy as np

from activation import sigmoid

class Agent():
    def __init__(self, n_inputs, n_outputs, genome=None, activation=sigmoid):
        if genome:
            self.genome = genome
        else:
            self.genome = Genome(n_inputs, n_outputs)
            
        self.network = Network(genome=self.genome, activation=activation)
        self.fitness = 0.0#sys.float_info.min
        
    def fitness_cal(self, fitness):
        return fitness(self.compiled_network)
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def more_complex(self, other):
        return len(self.genome.neurons) > len(other.genome.neurons)
    
    def getMaxValues(self, values):
        return np.max(values)
    
    def getMaxIndex(self, values):
        return int(np.argmax(values))
    
    
    
'''agent = Agent(n_inputs=3, n_outputs=1)

import cv2
width = 1000
height = 800

# Draw best Network
cv2.imshow("Agent Network", agent.network.draw(width, height))
cv2.waitKey(2)'''
