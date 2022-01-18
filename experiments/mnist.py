# -*- coding: utf-8 -*-

'''import keras
from keras.datasets import mnist

from parameter import params

# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
                                
print('x_train shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')


num_classes = 10
# convert class vectors to binary class matrices
#train_labels = keras.utils.to_categorical(train_labels, num_classes)
#test_labels = keras.utils.to_categorical(test_labels, num_classes)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

params['OUTPUTS'] = num_classes
params['INPUTS'] = train_images.shape[1]

from network import Network
from genome import Genome

genome = Genome(params['INPUTS'], params['OUTPUTS'])
network = Network(genome)

sample_1 = train_images[0]
#print(len(sample_1), sample_1.shape)
#print(sample_1)

res = network.getOutput(sample_1)
print(res)
print(train_labels[0])'''
training_data_file = open("../data/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

import numpy as np
np.random.seed(0)
from genome import Genome
from network import Network
from population import Population
from parameter import params

#record = training_data_list[0]
#record = record.split(',')
#inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
#targets = np.zeros(10) + 0.01
#targets[int(record[0])] = 0.99
#print(int(record[0]))

'''genome = Genome(784, 10)
network = Network(genome)
#res = network.getOutput(inputs)
#print(res)
#print("Max index ", res.index(max(res)))
epochs = 200
accuracies = []
losses = []
for epoch in range(20):
    print(">Epoch ", epoch)
    score_card = []
    L = 0
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99
            
        # get output
        outputs = network.getOutput(inputs)
        if outputs.index(max(outputs)) == int(all_values[0]):
            score_card.append(1)
        else:
            score_card.append(0)
            
        error_vector = []
        loss = 0
        for i in range(len(outputs)):
            error_vector.append(targets[i]-outputs[i])
            loss += error_vector[i]**2
        loss = loss/2
#        print("\tLoss ", loss)
        L += loss
        #backpropagate
        network.backpropagate(error_vector)
        
    accuracy = sum(score_card)/len(score_card)
    print("\tAccuracy ", accuracy)
    accuracies.append(accuracy)
    
    L = L/len(training_data_list)
    losses.append(L)
        
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(accuracies)
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.figure(2)
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')'''

     
params['NODE_MUTATE_PROB'] = 0.1 # 0.03
params['CONN_MUTATE_PROB'] = 0.3
params['pop_size'] = 20
params['gens'] = 10

params['INPUTS'] = 784
params['OUTPUTS'] = 10

def fitness(pop):
    for agent in pop.members:
        score_card = []
        for record in training_data_list[:1]:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
                    
            # get output
            outputs = agent.network.getOutput(inputs)
            if outputs.index(max(outputs)) == int(all_values[0]):
                score_card.append(1)
            else:
                score_card.append(0)
        
        agent.fitness = sum(score_card)
        
def fitness_learning(pop):
    for agent in pop.members:
        #epochs = np.random.randint(0,params['epochs'])
        epochs = 1
        for epoch in range(epochs):
            score_card = []
            L = 0
            for record in training_data_list[:1]:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(all_values[0])] = 0.99
                    
                # get output
                outputs = agent.network.getOutput(inputs)
                if outputs.index(max(outputs)) == int(all_values[0]):
                    score_card.append(1)
                else:
                    score_card.append(0)
                    
                error_vector = []
                loss = 0
                for i in range(len(outputs)):
                    error_vector.append(targets[i]-outputs[i])
                    loss += error_vector[i]**2
                loss = loss/2
                L += loss
                
                #backpropagate
                agent.network.backpropagate(error_vector)
                
            accuracy = sum(score_card)/len(score_card)
            L = L/len(training_data_list)
            
            agent.fitness = accuracy
            agent.loss = L
        
        
if __name__ == "__main__":
    pop = Population(pop_size=params['pop_size'], n_inputs=784, n_outputs=10)
    fitness_learning(pop)
    #fitness(pop)
    
    avg_fitness = []
    best_fitness = []
    best_loss = []
    species = []
    innovation = []
    
    for gen in range(params['gens']):
        print(">Gen ", gen, " Species: ", len(pop.species))
        species.append(len(pop.species))
        
        best = max(pop.members)
        best_fitness.append(best.fitness)
        best_loss.append(best.loss)
        avg_fitness.append(pop.compute_average_fitness())
        print(" Best fitness ", best.fitness, " Conn: ", \
              len(best.genome.connections), ", Neurons: ", len(best.genome.neurons))
            
        #Draw best Network
#        cv2.imshow("Network", best.network.draw(1000, 800))
#        cv2.waitKey(2)
            
        print("\t innovation number ", params['innov_no'])
        innovation.append(params['innov_no'])
        
        pop.evolve()
        
        fitness_learning(pop)
        #fitness(pop)
        
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

plt.figure(1)
plt.plot(best_loss)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.savefig('result/loss.png')
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


'''
# go through all records in the training data set
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    
    pass'''




