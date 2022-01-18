# -*- coding: utf-8 -*-

from neuron import Neuron, Layer
from connection import Connection
from activation import sigmoid

import copy
#import random
import numpy as np

from parameter import params

from collections import defaultdict

from helper import next_innov_no

class Genome:
    def __init__(self, n_inputs, n_outputs):
        self.connections = {}
        self.neurons = {}
        self.in_neurons = []
        self.out_neurons = []
        self.bias_neurons = []
        self.last_neuron = 0
        
        self.layers = 3
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs * 2
        
        self.create_genome()
    
    def clone(self):
        """
        Fast copy a genome
        """
        clone = Genome(self.n_inputs, self.n_outputs)
        clone.connections = copy.deepcopy(self.connections)
        clone.neurons = copy.deepcopy(self.neurons)
        clone.in_neurons = self.in_neurons
        clone.out_neurons = self.out_neurons
        clone.bias_neurons = self.bias_neurons
        clone.last_neuron = self.last_neuron
        clone.n_inputs = self.n_inputs
        clone.n_outputs = self.n_outputs
#        clone.fitness = self.fitness
        return clone
        
    def create_genome(self):
         # Create i/p and o/p neurons
        nid = 0
        for i in range(self.n_inputs):
            neuron = Neuron(layer=Layer.INPUT)
            neuron.id = nid
            self.neurons[nid] = neuron
            self.in_neurons.append(nid)
            nid += 1
            
        # double number of outputs for self-teaching
        for i in range(int(self.n_outputs/2)):
            neuron = Neuron(layer=Layer.OUTPUT)
            neuron.id = nid
            self.neurons[nid] = neuron
            self.out_neurons.append(nid)
            nid += 1
            
        for i in range(int(self.n_outputs/2)):
            neuron = Neuron(layer=Layer.TEACHING)
            neuron.id = nid
            self.neurons[nid] = neuron
            self.out_neurons.append(nid)
            nid += 1
            
        for i in range(params['BIAS']):
            neuron = Neuron(layer=Layer.BIAS)
            neuron.id = nid
            self.neurons[nid] = neuron
            self.bias_neurons.append(nid)
            nid += 1
            
        self.last_neuron = nid - 1
        # Create a gene for every ip, op pair
        innov_no = 0
        for i in range(self.n_inputs):
            for j in range(self.n_outputs):
                conn = Connection(innov_no=innov_no)
                conn.in_neuron = self.in_neurons[i]
                conn.out_neuron = self.out_neurons[j]
                conn.weight = np.random.uniform(params['w_min'], params['w_max'])
                self.connections[innov_no] = conn
                innov_no += 1
                
        for i in range(params['BIAS']):
            for j in range(self.n_outputs):
                conn = Connection(innov_no=innov_no)
                conn.in_neuron = self.bias_neurons[i]
                conn.out_neuron = self.out_neurons[j]
                conn.weight = np.random.uniform(params['w_min'], params['w_max'])
                self.connections[innov_no] = conn
                innov_no += 1
                
        
                
    def create_layers(self):
        neuronp = {x for x in self.in_neurons}
        first_layer = []
        for x in self.in_neurons:
            first_layer.append(x)
        layers = [first_layer]
        
        outputs = {x for x in self.out_neurons}
        last_layer = []
        for x in self.out_neurons:
            last_layer.append(x)
        
        #layers = [neuronp.copy()]
        remaining = {x for x in self.neurons.keys()} - neuronp - outputs
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        wt = {}
        for conn in self.connections.values():
            incoming[conn.out_neuron].append(conn.in_neuron)
            outgoing[conn.in_neuron].append(conn.out_neuron)
            wt[(conn.in_neuron, conn.out_neuron)] = [conn.weight, conn.enabled]
        while True:
            L = []
            for neuron in remaining:
                if set(incoming[neuron]) <= neuronp:
                    neuronp.add(neuron)
#                    for other_neuron in L:
#                        print("OTHER NEURON", other_neuron, " NEURON ", neuron)
                    L.append(neuron)

            if not L:
                break
            
            layers.append(L)
            for neuron in L:
                remaining.remove(neuron)
            
            if not remaining:
                if not outputs:
                    break
                else:
                    remaining = remaining | outputs
                    outputs = {}
                    

        return layers, incoming, outgoing, wt
    
    def compile_network(self, activation=sigmoid):
        layers, incoming, outgoing, wt = self.create_layers()

        def activate(inputs):
            # set the values for the inputs
            values = {x: 0.0 for x in self.neurons.keys()}
            for i, ip_n in enumerate(self.in_neurons):
                values[ip_n] = inputs[i]
            
            values[self.bias_neurons[0]] = 1.0
            
            for layer in layers[1:]:
                for neuron in layer:
                    total = 0
                    if not incoming[neuron]:
                        # if no incoming node, don't apply actv function
                        continue
                    for ip in incoming[neuron]:
                        total += wt[(ip, neuron)] * values[ip]
                    total = activation(total)
                    values[neuron] = total

            outputs = [values[op] for op in self.out_neurons]
            return outputs

        return activate
        
    def mutate(self):
        """
        Given a genome, mutates it in-place
        """
        NODE_MUTATE_PROB = params['NODE_MUTATE_PROB']
        CONN_MUTATE_PROB = params['CONN_MUTATE_PROB']
        WT_MUTATE_PROB = params['WT_MUTATE_PROB']
        WT_PERTURBED_PROB = params['WT_PERTURBED_PROB']
        
        if np.random.random() < NODE_MUTATE_PROB:
            self.mutate_add_node()
        if np.random.random() < CONN_MUTATE_PROB:
            self.mutate_add_conn()
        if np.random.random() < WT_MUTATE_PROB:
            for conn in self.connections.values():
                conn.mutate(WT_PERTURBED_PROB)
                
    def mutate_add_conn(self):
        # Select any 2 neurons
        # If they are not connected, connect them
        # Make sure that the the op neuron is not
        # from an input layer
        n1 = np.random.choice([x for x in self.neurons.values() \
                               if x.layer != Layer.OUTPUT \
                               and x.layer != Layer.TEACHING])
        n2 = np.random.choice([x for x in self.neurons.values() \
                if x.layer != Layer.INPUT and x.layer != Layer.BIAS])
        
        nid1 = n1.id
        nid2 = n2.id
        if nid1 == nid2:
            return
        # check if a cyclic link exists
        if self.detect_cycle(nid1, nid2):
            return
        if set([(nid1, nid2), (nid2, nid1)]) & set([(x.in_neuron, x.out_neuron) for x in self.connections.values()]):
            return

        innov_no = next_innov_no()
        conn = Connection(in_neuron=nid1, out_neuron=nid2, weight=1.0, innov_no=innov_no)
        self.connections[innov_no] = conn
        
    def detect_cycle(self, ip, op):
        if ip == op:
            return False
        
        incoming = defaultdict(list)
        for conn in self.connections.values():
            incoming[conn.out_neuron].append(conn.in_neuron)
            
        unexplored = set([ip])
        explored = set()
        while unexplored:
            node = unexplored.pop()
            explored.add(node)
            for n in incoming[node]:
                if n not in explored:
                    unexplored.add(n)
            if op in explored:
                return True
        return False
    
    def mutate_add_node(self):
        # Select any conn
        # Split it into two connections
        '''conn_list = list(self.connections.values())
        enabled_list = []
        for conn in conn_list:
            if conn.enabled:
                enabled_list.append(conn)
        if len(enabled_list) == 0:
            return
        else:
            conn =  np.random.choice(enabled_list)'''
        conn = np.random.choice(list(self.connections.values()))
        if not conn.enabled:
            return
        conn.enabled = False

        ip, op, wt = conn.in_neuron, conn.out_neuron, conn.weight
        neuron = Neuron(layer=Layer.HIDDEN)
        nid = self.next_nid()
        if ip == op or ip == nid or nid == op:
            return
        neuron.id = nid
        self.neurons[nid] = neuron
        
        innov_no1 = next_innov_no()
        innov_no2 = next_innov_no()
        conn1 = Connection(in_neuron=ip, out_neuron=nid, weight=1.0, enabled=True, innov_no=innov_no1)
        conn2 = Connection(in_neuron=nid, out_neuron=op, weight=wt, enabled=True, innov_no=innov_no2)
        self.connections[innov_no1] = conn1
        self.connections[innov_no2] = conn2
        
    def next_nid(self):
        """
        Tracker for next neuron id in the given genome
        """
        nid = self.last_neuron + 1
        self.last_neuron = nid
        return nid
    
    @staticmethod
    def crossover(mom, dad):
        """
        Mates 2 individuals and returns an offspring
        mom should be the fitter one
        """
        
        # create a new child and copy over all the 
        # information except for genes
        # from mom (the fitter parent)
        child = mom.clone()
        child.connections = {}
        
        # Copy genes from both parents to the child
        # We use historical markings i.e. the innovation 
        # numbers (which are keys of the genes dict)
        for conn in mom.connections:
            if conn in dad.connections and np.random.random() < 0.5:
                # matching gene is copied from either parents with a probability
                child.connections[conn] = copy.copy(dad.connections[conn])
            else:
                # disjoint gene, copy from the fitter parent
                child.connections[conn] = copy.copy(mom.connections[conn])

        return child
    
    @staticmethod
    def calc_DEW(g1, g2):
        conn1_set = {x for x in g1.connections}
        conn2_set = {x for x in g2.connections}
        excess_marker = max(conn1_set)
        
        complete = conn1_set | conn2_set
        matching = conn1_set & conn2_set
        avg_wt = 0
        for conn in matching:
            avg_wt += abs(g1.connections[conn].weight - g2.connections[conn].weight)
        avg_wt /= len(matching)
        
        non_matching = complete - matching
        excess = len([x for x in non_matching if x > excess_marker])
        disjoint = len([x for x in non_matching if x <= excess_marker])
        return disjoint, excess, avg_wt
    
    @staticmethod
    def computeCompabilityDistance(g1, g2):
        """
        computes compatability distance between two genomes.
        delta = c1*excess/N + c2*disjoint/N + c3*avg_weights
        N = maximum length between g1 and g2
        """
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        N = max(len(g1.connections), len(g2.connections))
        N = 1 if N < 20 else N
        d, e, w = Genome.calc_DEW(g1, g2)
        delta = (c2 * d + c1 * e)/N + c3 * w
        return delta

        
    