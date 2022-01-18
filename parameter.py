# -*- coding: utf-8 -*-
import activation

params = {
        # Neural Network
        'w_min': -1,
        'w_max': 1,
        'INPUTS': 2,
        'OUTPUTS': 1,
        'BIAS': 1,
        
        # learning rate
        'lr': 0.01,
        'epochs': 20,
        
        # Evolutionary
        'pop_size': 50,
        'gens': 300,
        # mutation
        'NODE_MUTATE_PROB': 0.1, # 0.03
        'CONN_MUTATE_PROB': 0.3, # larger species, 0.3 prob of adding a new link mutation
        'WT_MUTATE_PROB': 0.8,
        'WT_PERTURBED_PROB': 0.9,
        
        # crossover
        'no_crossover': 0.25,
        'inter_species': 0.001,
        
        # innovation number
        'innov_no': 0,
        
        # speciation
        'c1': 1.0,
        'c2': 1.0,
        'c3': 3.0, # 3.0 for pole balancing
        'compabilityThreshold': 3.0,
        
        # max fitness
        'max_fitness': 4,
        
        # activation
        'input_act_fn': activation.linear,
        'hidden_act_fn': activation.relu,
        'output_act_fn': activation.sigmoid,
        'bias_act_fn': activation.linear,
        
    }