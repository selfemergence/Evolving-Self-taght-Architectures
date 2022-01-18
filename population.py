# -*- coding: utf-8 -*-
#from genome import Genome
from agent import Agent
from species import Species
from parameter import params

from activation import sigmoid

class Population:
    def __init__(self, pop_size, n_inputs, n_outputs, activation=sigmoid):
        self.pop_size = pop_size
        self.members = [Agent(n_inputs=n_inputs, n_outputs=n_outputs, \
                              activation=activation) for _ in range(pop_size)]
        
        # starting innovation number
        params['innov_no'] = len(self.members[0].network.connections)
        
        self.species = []
        
        self.champion = None
        self.maxFitness = 0
        
    def size(self):
        return len(self.members)
    
    def get_member(self, index):
        return self.mmembers[index]
        
    def speciate(self):
        """Speciate the population"""
        #clear all Species
        for species in self.species:
            species.clear()
            
        #add every member to species
        for member in self.members:
            #compare to all available Species
            speciesFound = False
            for species in self.species:
                if species.sameSpecies(member):
                    species.addMember(member)
                    speciesFound = True
                    break
                
            #no matching Species found
            if not speciesFound:
                #create a new Species
                self.species.append(Species(member))
                
    def compute_average_fitness(self):
        return sum([network.fitness for network in self.members])/len(self.members)
                
    #----------------------------------------------------------------------------------

    def sortSpecies(self):

        #sort members inside each Species
        for species in self.species:
            species.sort()

        #sort Species
        sorted = []
        while len(self.species) > 0:

            #find maximum
            maxIndex = 0
            max = self.species[0].bestFitness
            for i in range(len(self.species)):
                if self.species[i].bestFitness > max:
                    max = self.species[i].bestFitness
                    maxIndex = i
            
            #remove max and add to sorted
            sorted.append(self.species.pop(maxIndex))
            
        self.species = sorted

        #if new best Player
        if self.species[0].bestFitness > self.maxFitness:
            self.champion = self.species[0].champion
            self.maxFitness = self.champion.fitness
        
    #----------------------------------------------------------------------------------

    def killBadMembers(self):
        """Species must be sorted"""

        #compute sum over all average Fitnesses
        avgSum = 0
        for species in self.species:
            #compute avg Fitnesses
            species.computeAvgFitness()
            #sum up avgFitness
            avgSum += species.avgFitness
                        
        #process Species always keeping the best one
        for i in range(len(self.species) - 1, 0, -1):

            #kill Stagnant species, i.e. no improvement after 15 generations
            if self.species[i].stagnancy >= 15:
                self.species.pop(i)
                continue

            #kill bad Species
#            elif self.species[i].avgFitness/avgSum * self.pop_size < 1:
#                self.species.pop(i)
#                continue

            #cull all Species
            else:
                self.species[i].cull()
        
        #re-Update fitness values
        for species in self.species:
            
            #species.fitnessSharing()
            species.computeAvgFitness()
            
    #----------------------------------------------------------------------------------

    def reproduce(self):

        avgSum = 0
        for species in self.species:
            avgSum += species.avgFitness
            
        # sort the population
        self.members = sorted(self.members, reverse=True)
        
        children = []
        # add elitist agents
        for i in range(params['elitism']):
            children.append(self.members[i])
        
        for species in self.species:
            #add each species champion
            children.append(species.champion)
            #compute number of children from current Species
            noChildren = int(species.avgFitness / avgSum * self.pop_size) - 1
            #create children
            for _ in range(noChildren):
                children.append(species.createChild())

        #fill with children of best species
        while len(children) < self.pop_size:
            children.append(self.species[0].createChild())

        self.members = children
        
    #--------------------------------------------------------------------------------
    def evolve(self):
        """evolution"""
        self.speciate()
        self.sortSpecies()
        self.killBadMembers()
        
        #reproduction
        self.reproduce()
        
        
    