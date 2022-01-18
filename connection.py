# -*- coding: utf-8 -*-

#import random
import numpy as np
from parameter import params

class Connection:
    global_innovation = 0
    
    def __init__(self, in_neuron=None, out_neuron=None, weight=0.0, enabled=True, innov_no=0):
        """
        Create a simple base gene i.e. a connection/synapse
        """
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.weight = weight
        self.enabled = enabled
        self.innovation = Connection.createInnovationNumber() if innov_no == None else innov_no
        
    def mutate(self, mutationRate):
        #mutate Weight
        if np.random.random() < mutationRate:
            #90% chance of tweeking the value a bit
            pertubationValue = (np.random.uniform(params['w_min'], params['w_max']))/10
            self.weight += pertubationValue
        else:
            #10% chance of changing Weight completely
            self.weight = np.random.uniform(params['w_min'], params['w_max'])
            
        #keep weight in between bounds
#        self.weight = min(1, max(-1, self.weight))
        
    def createInnovationNumber():
        #get current innovation Number
        curInnovationNumber = Connection.globalInnovationNumber
        #increase global innovation Number
        Connection.globalInnovationNumber += 1
        #return current innovation Number
        return curInnovationNumber
    
    def __repr__(self):
        string = "Connection {}: w = {} ({} -> {}) {}"
        return string.format(self.innovation, self.weight, self.in_neuron, self.out_neuron,
                             "Enabled" * self.enabled or "Disabled")