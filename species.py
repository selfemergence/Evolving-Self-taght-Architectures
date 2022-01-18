# -*- coding: utf-8 -*-
from genome import Genome
from agent import Agent
from parameter import params
#import random
import numpy as np

class Species:
    def __init__(self, agent):
        #members
        self.members = [agent]
        self.champion = agent
        self.repr = agent.genome
        
        #Fitness
        self.bestFitness = agent.fitness
        self.avgFitness = 0
        #generations without improvement
#        self.stagnant = 0
#        self.staleness = 0
        self.stagnancy = 0
        
    #----------------------------------------------------------------------------------

    def clear(self):
        self.members = []
        
    #----------------------------------------------------------------------------------
    def addMember(self, agent):
        self.members.append(agent)
        
    #----------------------------------------------------------------------------------

    def sameSpecies(self, agent):

        compabilityThreshold = params['compabilityThreshold']
        #compare representative with given Genom
        compability = Genome.computeCompabilityDistance(self.repr, agent.genome)
        #compare to Threshold
        return compabilityThreshold > compability
            
    #----------------------------------------------------------------------------------
            
    def computeAvgFitness(self):

        if len(self.members) == 0:
            self.avgFitness = 0
            return

        sum = 0
        for member in self.members:
            sum += member.fitness

        self.avgFitness = sum / len(self.members)
        
    #----------------------------------------------------------------------------------

    def fitnessSharing(self):

        #devide all fitnesses by size of Species
        for member in self.members:
            member.fitness /= len(self.members)
            
    #----------------------------------------------------------------------------------

    def cull(self):
        """loose bottom half of Species"""

        if len(self.members) > 2:
            self.members = self.members[:len(self.members)//2]
            
    def update_repr(self):
        """update representation"""
        self.repr = self.members[0].genome.copy()
        
        
    #----------------------------------------------------------------------------------

    def sort(self):
        #Sort Species
        temp = []

        #Selection Sort Algorithm
        while len(self.members) != 0:
            
            #find maximum
            maxIndex = 0
            max = self.members[0].fitness
            for j in range(len(self.members)):
                if self.members[j].fitness > max:
                    max = self.members[j].fitness
                    maxIndex = j
            
            #remove max and add to temp
            temp.append(self.members.pop(maxIndex))

        self.members = temp

        #is there a new best player
        if len(self.members) > 0 and self.members[0].fitness > self.bestFitness:
            self.staleness = 0
            self.champion = self.members[0]
            self.bestFitness = self.champion.fitness
            self.repr = self.champion.genome
        else:
            self.stagnancy += 1
            
    #----------------------------------------------------------------------------------

    def selectMember(self):
        '''Fitness-proportionate Selection
        '''

        #sum all fitnesses
        fitnessSum = 0
        for member in self.members:
            fitnessSum += member.fitness

        #create random number
        rand = np.random.uniform(0, fitnessSum)

        #finding member
        runningSum = 0
        for member in self.members:
            runningSum += member.fitness

            if runningSum > rand:
                return member
        return self.members[0]
    
    def tournament_selection(self, tour_size=8):
        if len(self.members) > 8:
            competitors = np.random.choice(self.members, tour_size)
        else:
            competitors = self.members
        return max(competitors)
            
    #----------------------------------------------------------------------------------

    def createChild(self):

        if np.random.uniform(0, 1) < params['no_crossover']:
#            chance of creating a child only by Mutation
            childGenome = self.selectMember().genome.clone()
            
        else:

        #get parents for CrossOver
            parentA = self.tournament_selection()
            parentB = self.tournament_selection()
        
            #create child by Crossover
#            if parentA.__eq__(parentB):
#                if not parentA.more_complex(parentB):
#                    parentA, parentB = parentB, parentA
            if parentA.__lt__(parentB):
                parentA, parentB = parentB, parentA
                
            childGenome = Genome.crossover(parentA.genome, parentB.genome)
                
            childGenome.n_inputs = parentA.genome.n_inputs
            childGenome.n_outputs = parentA.genome.n_outputs
        
        #Mutate Child
        newChildGenome = childGenome.clone()
        newChildGenome.mutate()

        #create Phenotyp of child Genom
        child = Agent(genome=newChildGenome,
                      n_inputs=childGenome.n_inputs,
                      n_outputs=childGenome.n_outputs)
        
        
        return child

            
        
        
        
 
        
    
