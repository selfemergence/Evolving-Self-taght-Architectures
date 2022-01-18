# -*- coding: utf-8 -*-
from parameter import params
from collections import defaultdict


def next_innov_no():
    """
    Tracker for global innovations among genes
    """
#    global innov_no
#    innov_no += 1
    #generation['innov_no'] += 1
#    return innov_no
    params['innov_no'] += 1
    return params['innov_no']

def compute_delta(neuron_id, network):
    """
    Compute Delta for a neuron with neuron_id in the network
    #######
    - If i is the output neuron:
        ##### delta[i] = derivative(activation[i]) * error[i]####3
    - If i is the hidden neuron:
        ##### delta[i] = derivative(activation[i]) * (Sum_over_j(delta[j]*activation[j]*w[i][j]))
    #######
    """
    if network.neurons[neuron_id].layer == 2: # Output
        
        # satuation due to sigmoid - first term
        activation = network.neuron_activation[neuron_id]
        derivative = (1 - activation) * activation
        
        # error term
        error_term = network.error[network.out_neurons.index(neuron_id)]
        
        # calculate delta
        network.delta[neuron_id] = derivative * error_term
        
        #print("Delta for neuron ", neuron_id, " is ", network.delta[neuron_id])

        
    elif network.neurons[neuron_id].layer == 1: # Hidden
        
        # satuation due to sigmoid - first term
        activation = network.neuron_activation[neuron_id]
        derivative = (1 - activation) * activation
        
        # sum over neuron post_id connecting/outgoing from neuron_id
        sum_over = 0
        for post_id in network.outgoing[neuron_id]:
            print("\tPost neuron ", post_id, " connects to neuron ", neuron_id)
            delta_term = network.delta[post_id]
            print("\tDelta of neuron ", post_id, " is ", delta_term)
            activation_term = network.neuron_activation[post_id]
            weight_term = network.weights[(neuron_id, post_id)]
            sum_over += delta_term * activation_term * weight_term
        
        network.delta[neuron_id] = derivative * sum_over
        #print("Delta for neuron ", neuron_id, " is ", network.delta[neuron_id])
        
    print("Delta for neuron ", neuron_id, " is ", network.delta[neuron_id])
        
    pass