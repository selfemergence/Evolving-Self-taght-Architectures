# -*- coding: utf-8 -*-

from genome import Genome
from collections import defaultdict
import copy
from parameter import params

from neuron import Layer

import cv2
import numpy as np


from activation import sigmoid, tanh, sigmoid1

class Network:
    def __init__(self, genome, activation=tanh):
#        self.genome = genome
#        self.genome.create_genome()
#        genome.create_genome()

        self.connections = copy.deepcopy(genome.connections)
        self.neurons = copy.deepcopy(genome.neurons)
        self.in_neurons = copy.deepcopy(genome.in_neurons)
        self.out_neurons = copy.deepcopy(genome.out_neurons)
        self.bias_neurons = copy.deepcopy(genome.bias_neurons)
        
        self.n_inputs = genome.n_inputs
        self.n_outputs = genome.n_outputs
        
        self.activation = activation
        
        self.layers, self.incoming, self.outgoing, self.weights = genome.create_layers()
        
        self.delta = defaultdict(dict)
        
        #sort nodes by layers
        self.nodes = [[] for i in range(genome.layers+1)]
        #loop over all nodes
        for node in self.neurons.values():
            if node.layer != Layer.BIAS:
                self.nodes[node.layer].append(node)
            else:
                self.nodes[-1].append(node)
        
    def getOutput(self, inputs):
        layers, incoming, wt = self.layers, self.incoming, self.weights
            
#        inputs = np.array(inputs, ndmin=2).T
#        print("\nInput ", type(inputs), inputs)
        # set the values for the inputs
        self.neuron_excitation = defaultdict(dict)
        
        values = {x: 0.0 for x in self.neurons.keys()}
        for i, ip_n in enumerate(self.in_neurons):
            values[ip_n] = inputs[i]
            self.neuron_excitation[ip_n] = inputs[i]
        
        values[self.bias_neurons[0]] = 1.0
        
        for layer in layers[1:]:
            for neuron in layer:
                total = 0
                if not incoming[neuron]:
                    # if no incoming node, don't apply actv function
                    total += 0
                    #continue
                else:
                    for ip in incoming[neuron]:
                        if wt[(ip, neuron)][1]:
                            #print(type(wt[(ip, neuron)][0]), type(values[ip]))
                            total += wt[(ip, neuron)][0] * values[ip]
                        else:
                            total += 0
                        
                self.neuron_excitation[neuron] = total
                #total = self.activation(total)
                total = self.neurons[neuron].activation(total)
                values[neuron] = total
                
        self.neuron_activation = copy.deepcopy(values)

        outputs = [values[op] for op in self.out_neurons][:int(self.n_outputs/2)]
        teaching_outputs = [values[op] for op in self.out_neurons][int(self.n_outputs/2):]
#        print("\nOutput ", outputs.shape, outputs)
        self.outputs = outputs
        self.teaching_outputs = teaching_outputs
        return outputs
    
    def compute_delta(self, neuron_id):
        """
        Compute delta for each neuron in the network
        #######
        + If i is the output neuron:
        ##### delta[i] = derivative(activation[i]) * error[i]####3
        + If i is the hidden neuron:
        ##### delta[i] = derivative(activation[i]) * (Sum_over_j(delta[j]*activation[j]*w[i][j]))
        """
        if self.neurons[neuron_id].layer == 2: # Output
        
            # satuation due to sigmoid - first term
            activation = self.neuron_activation[neuron_id]
            derivative = self.neurons[neuron_id].derivative(activation)
            
        
            # error term
            error_term = self.error[self.out_neurons.index(neuron_id)]
        
            # calculate delta
            self.delta[neuron_id] = derivative * error_term
        
            #print("Delta for neuron ", neuron_id, " is ", network.delta[neuron_id])

        
        elif self.neurons[neuron_id].layer == 1: # Hidden
        
            # satuation due to sigmoid - first term
            activation = self.neuron_activation[neuron_id]
            derivative = self.neurons[neuron_id].derivative(activation)
            #self.activation(activation, True)#(1 - activation) * activation
            
            # sum over neuron post_id connecting/outgoing from neuron_id
            sum_over = 0
            for post_id in self.outgoing[neuron_id]:
                #print("\tPost neuron ", post_id, " connects to neuron ", neuron_id)
                delta_term = self.delta[post_id]
                #print("\tDelta of neuron ", post_id, " is ", delta_term)
                activation_term = self.neuron_activation[post_id]
                weight_term = self.weights[(neuron_id, post_id)][0]
                sum_over += delta_term * activation_term * weight_term
                
            self.delta[neuron_id] = derivative * sum_over
            #print("Delta for neuron ", neuron_id, " is ", network.delta[neuron_id])
        
        #print("Delta for neuron ", neuron_id, " is ", self.delta[neuron_id])
    
    def backpropagate(self, error):
        """
        1> At the output layer:
        Delta(w[i][j]) = (o[i] - t[i])*(1-o[i])*o[i]*a[j]
        ==> w[i][j] += alpha* Delta(w[i][j])
        
        - w[i][j] = weights from neuron j to neuron i
        - a[i] = activation of neuron i (same as o[i])
        - o[i] = output of neuron i at the output layer
        - t[i] = target value at neuron i
        - alpha = learning rate
        - x[i] = excitation of neuron i ((sum of weighted activations coming into neuron i, before squashing)
        
        2> At the hidden layer:
        Delta(w[i][j] = - (Sigma(d[k]w[ki])) * (1-a[i])a[i] * a[j]
        - i = neuron in the hidden layer
        - j = neuron in the previous layer (maybe input layer) connecting to neuron i
        - the hidden nodes do not themselves make errors, rather they
        contribute to the errors of the output nodes.
        - So, the derivative of the total output error w.r.t. a hidden neuron’s activation is the sum of that
        hidden neuron’s contributions to the errors in all of the output neurons:
        ==> Credit Assignment Problem
        
        ########
        w[i][j] = w[i][j] + lr * delta[i] * activation[j]
        ########
        """
        for layer_ff in reversed(self.layers):
            for neuron_id in layer_ff:
                
                # check if the neuron is output neuron
                if self.neurons[neuron_id].layer == 2:
                    
                    # saturation due to sigmoid - first term
                    activation  = self.neuron_activation[neuron_id]
                    derivative = self.neurons[neuron_id].derivative(activation)#(1 - activation) * activation
                    
                    # error term
                    error_term = error[self.out_neurons.index(neuron_id)]
                    
                    # calculate delta
                    #print(type(derivative), type(error_term))
                    self.delta[neuron_id] = derivative * error_term
                    
                    # change the weight connecting to neuron_id
                    for pre_id in self.incoming[neuron_id]:
                        if self.weights[(pre_id, neuron_id)][1]:
                            self.weights[(pre_id, neuron_id)][0] += params['lr'] * self.delta[neuron_id] \
                                                            * self.neuron_activation[pre_id]
                
                # or else if the neuron is hidden
                elif self.neurons[neuron_id].layer == 1:
                    
                    # saturation due to sigmoid - first term
                    activation  = self.neuron_activation[neuron_id]
                    derivative = self.neurons[neuron_id].derivative(activation)#(1 - activation) * activation
                    
                    # sum over neuron post_id connecting/outgoing from neuron_id
                    sum_over = 0
                    for post_id in self.outgoing[neuron_id]:
                        # sum = delta_term * activation_term * weight_term
                        
                        #check if post_id and neuron_id in the same layer
                        if post_id in layer_ff:
                            delta_term = 0
                        elif self.neurons[post_id].layer==Layer.TEACHING:
                            delta_term = 0
                        else:
                            delta_term = self.delta[post_id]
                        activation_term = self.neuron_activation[post_id]
                        weight_term = self.weights[(neuron_id, post_id)][0]
                        
                        #### DEBUG
                        if delta_term == {}:
                            print("ERROR")
                            print(post_id, self.neurons[post_id])
                            print(neuron_id, self.neurons[neuron_id])
                            print(self.layers)
                            print("Incoming ", self.incoming)
                            print("Outgoing ", self.outgoing)
                            print("Weights ", self.weights)
                            
                        ### END DEBUG
                        
                        sum_over += delta_term * activation_term * weight_term
                    
                    # calculate delta for neuron_id
                    self.delta[neuron_id] = derivative * sum_over
                    
                    # change the weight for incoming connections to neuron_id
                    for pre_id in self.incoming[neuron_id]:
                        if self.weights[(pre_id, neuron_id)][1]:
                            self.weights[(pre_id, neuron_id)] += params['lr'] * self.delta[neuron_id] \
                                                           * self.neuron_activation[pre_id]
                        
        pass
    
    def self_taught(self):
        error_vector = []
        for i in range(len(self.teaching_outputs)):
            error_vector.append(self.teaching_outputs[i] - self.outputs[i])
        self.backpropagate(error_vector)
                
        
    #----------------------------------------------------------------------------------
    
    def draw(self, width, height):

        #empty Image
#        img = np.zeros((height, width))
        
        img = np.ones((height,width,3), np.uint8)
        color = [0, 0, 102]
        img[:] = color

        #Node Position Dictionary
        NodePositions = {}
        #Compute Draw Position of each Neuron
        for layerIdx in range(len(self.nodes)):
            for nodeIdx in range(len(self.nodes[layerIdx])):

                NodePositions[self.nodes[layerIdx][nodeIdx]] = (
                    int(20 + (width-40) * (layerIdx/(len(self.nodes) - 1))),
                    int((height-40) / (len(self.nodes[layerIdx]) + 1) * (nodeIdx + 1))
                    )
        #Draw Node Points
#        for pos in NodePositions.values():
#            cv2.circle(img, pos, 5, (1))

        #Draw Connections
        for connection in self.connections.values():
            
            if connection.enabled: #and connection.weight != 0:
               
                in_neuron = self.neurons[connection.in_neuron]
                out_neuron = self.neurons[connection.out_neuron]
                pointA = NodePositions[in_neuron]
                pointB = NodePositions[out_neuron]
                    
                cv2.line(img, pointA, pointB, (0,255,0), 2)
                
            else:
#                print("Draw connection ", connection)
                pointA = NodePositions[self.neurons[connection.in_neuron]]
                pointB = NodePositions[self.neurons[connection.out_neuron]]
                    
                cv2.line(img, pointA,pointB, (255,0,0), 1)
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for neuron in self.neurons.values():
            point = NodePositions[neuron]
            label = ""
            color = []
            if neuron.layer == 0:
                label = "Input"
                x = point[0] + 20
                y = point[1] + 20
                color = [0, 255, 255]
            elif neuron.layer == 1:
                label = "Hidden"
                x = point[0] - 20
                y = point[1] - 20
                color = [0, 255, 0]
            elif neuron.layer == 2:
                label = "Output"
                y = point[1] - 40
                x = point[0] - 40
                color = [0, 0, 255]
            elif neuron.layer == 3:
                label = "Teaching"
                y = point[1] - 40
                x = point[0] - 40
                color = [0, 0, 255]
            elif neuron.layer == 4:
                label = "Bias"
                y = point[1] - 40
                x = point[0] - 40
                color = [255, 255, 255]
            cv2.putText(img, label, (x, y), font, 0.5, (200,255,155), 1, cv2.LINE_AA)
            
            cv2.circle(img, point, 10, color, -1)
                
        return img
        

### Test #####
#import numpy as np
#np.random.seed(0)    
#width = 1000
#height = 800
#
#
#genome = Genome(2, 2)
#for i in range(1):
#    genome.mutate()
#network = Network(genome)

#for neuron in network.neurons.values():
#    print(neuron)
#for conn in network.connections.values():
#    print(conn)
#
#inputs = (1, 0)
#outputs = network.getOutput(inputs)
#
#
#
#for key, value in network.neuron_activation.items():
##    print("Activation of neuron ", key, " is ", value)
#    pass
#    
#for key, value in network.neuron_excitation.items():
##    print(key, value, sigmoid(value))
#    pass
#
#print("Layers ", network.layers)
#print("Incomings ", network.incoming)
#print("Weights ", network.weights)
#
#  
#target = (1, 1)
##network.error = [target[0]-outputs[0], target[1] - outputs[1]]
##print("Error = ",  network.error)
#
#
#errors = []
#steps = 150
#
#for i in range(steps):
#    outputs = network.getOutput(inputs)
#    error = [target[0]-outputs[0], target[1] - outputs[1]]
#    errors.append(1/2*(error[0]**2 + error[1]**2))
#    #network.backpropagate(error)
#    network.self_taught()
#
#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.plot(errors)
#plt.xlabel("Learning steps")
#plt.ylabel("Loss")


# Draw best Network
#cv2.imshow("Agent Network", network.draw(width, height))
#cv2.waitKey(0)

'''
#empty Image
img = np.zeros((height, width))

#Node Position Dictionary
NodePositions = {}
#Compute Draw Position of each Neuron
for layerIdx in range(len(network.nodes)):
    for nodeIdx in range(len(network.nodes[layerIdx])):
        NodePositions[network.nodes[layerIdx][nodeIdx]] = (
                int(20 + (width-40) * (layerIdx/(len(network.nodes) - 1))),
                int((height-40) / (len(network.nodes[layerIdx]) + 1) * (nodeIdx + 1))
                )
        
#for index, node in NodePositions.items():
#    print(index, node)
    
#for connection in network.connections.values():
#    pointA = network.neurons[connection.in_neuron]
#    print(pointA)'''
    