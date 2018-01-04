# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 12:16:22 2017

@author: Chettomastiksilikon
"""

import random
import numpy as np
import sys

def read_problem(file_name):
    lines = open(file_name).readlines()

    # get the number of items and total capacity
    tokens = lines[0].split()
    n_items = int(tokens[0])
    capacity = int(tokens[1])

    # collect the values and weights of each item
    items = []
    for line in lines[1 : n_items+1]:
        tokens = line.split()
        value = int(tokens[0])
        weight = int(tokens[1])
        item = (value, weight)
        items.append(item)

    return (capacity, items)

def randInitializeMaxCapacity(items, capacity):
    tempTaken = np.zeros((len(items)))
    tempWeight = 0
    tempValue = 0
    for i in range(0, len(items)):
        zerosIndexes = np.where(tempTaken == 0)[0]
        chosenZero = random.randint(0, len(zerosIndexes)-1) 

        newWeight = items[zerosIndexes[chosenZero]][1] + tempWeight
        newValue = items[zerosIndexes[chosenZero]][0] + tempValue
        if(newWeight < capacity):
            tempWeight = newWeight
            tempValue = newValue   
            tempTaken[zerosIndexes[chosenZero]] = 1
        else:
            break
    
    return (tempTaken, tempWeight, tempValue)

def sackOptimizer(mutatedChild, opt, delHeuristic):
    items2 = np.array(items)
    mutatedChildA1 = [int(i) for i in mutatedChild]
    mutatedChildA1 = np.array(mutatedChildA1)
    total_weight = wFitness(mutatedChild)
    
    while(total_weight > c):
        onesIndexes = np.where(mutatedChildA1 == 1)[0]
        
        if(random.random() > delHeuristic):
            chosenOne = onesIndexes[np.argmin(items2[onesIndexes,0] / items2[onesIndexes,1])]
        else:
            chosenOne = onesIndexes[random.randint(0,len(onesIndexes)-1)]
        mutatedChildA1[chosenOne] = 0
        
        total_weight = total_weight - items2[chosenOne,1]
        
    #improvement section
    if(random.random() > opt):
        maximum_increase = 0
        onesIndexes = np.where(mutatedChildA1 == 1)[0]
        fixedOnes = onesIndexes.copy()
        zerosIndexes = np.where(mutatedChildA1 == 0)[0]
        temp_weight = total_weight
        to_one = 0
        to_zero = 0
        
        for i in fixedOnes:
            maximum_increase = 0
            for j in zerosIndexes:
                value_change = items2[j,0] - items2[i,0]
                temp_weight = total_weight + items2[j,1] - items2[i,1]
                if(value_change > maximum_increase and temp_weight <= c):
                    maximum_increase = value_change
                    to_one = j
                    to_zero = i
            if(maximum_increase > 0):
                total_weight = total_weight + items2[to_one,1] - items2[to_zero,1]
                mutatedChildA1[to_one] = 1
                mutatedChildA1[to_zero] = 0
            zerosIndexes = np.where(mutatedChildA1 == 0)[0]           

    return list(mutatedChildA1)

def mutate(human):
    """
    Takes a binary list and flips bits at a probability of pM, outputs another binary list.
    """
    xman = human[:]
    for i in range(n):
        if pM > random.random():
            if human[i] == 0:
                xman[i] = 1
            else:
                xman[i] = 0
    return xman

def unifXover(parentA, parentB):
    """
    Takes 2 binary lists and with probablity pX performs uniform crossover at probability pU to produce a list of 2 new binary lists.
    """
    childA = parentA[:]
    childB = parentB[:]
    if pX > random.random():
        for i in range(n):
            if pU > random.random():
                childA[i] = parentB[i]
                childB[i] = parentA[i]
    return [childA, childB]
        
def result_information(b):
    """
    Accepts a binary list denoting packed items and returns a list of their index numbers, total value and total weight.
    """
    total_value = 0
    total_weight = 0

    for i in range(n):
        if(b[i] == 1):
            total_value += items[i][0]
            total_weight += items[i][1]
    return total_value, total_weight

def vFitness(b):
    """
    Accepts a binary list denoting packed items and returns their total value.
    """
    total_value = 0
    for i in range(n):
        if(b[i] == 1):
            total_value += items[i][0]
 
    return total_value

def wFitness(b):
    """
    Accepts a binary list denoting packed items and returns their total weight.
    """
    total_weight = 0
    for i in range(n):
        if(b[i] == 1):
            total_weight += items[i][1]
            
    return total_weight

def tournament_selection(pop, K):
    """
    Takes population list of binary lists and tournament size and returns a winning binary list.
    """
    tBest = 'None'
    for i in range(K):
        contestant = pop[random.randint(0, P-1)]
        if (tBest == 'None') or vFitness(contestant) > vFitness(tBest):
            tBest = contestant[:]
    return tBest

def initialize_population():
    """
    Generates a list of binary lists each representing a valid packing selection.
    """
    inits = []
    for i in range(P):
        tempTaken, tempWeight, tempValue = randInitializeMaxCapacity(items, c)   
        inits.append(list(tempTaken))
    return inits

def select_elites():
    """
    Selects the E best solutions from the population and returns them as a list of binary lists.
    """
    elites = []
    while len(elites) < E: # Keep choosing elites until there are E of them
        new_elites = popD[max(popD)] # These are the binary lists with the best fitness
        # If adding all the elites with this fitness would be too many, then discard the surplus at random
        while len(new_elites) > (E - len(elites)):
            new_elites.remove(random.sample(new_elites, 1)[0])
        elites.extend(new_elites)
        popD.pop(max(popD), None) # Remove the key with the value just added from popD{}
    return elites

def updateBest():
    return [max(popD), popD[max(popD)][0], s]

def rankedList():
    # Make a list of each binary list and its fitness
    return [(vFitness(i), i) for i in popL]
   
def rankedDict():
    # Make a dictionary where keys are fitness and values are tuples of the binary lists with that fitness
    popD = {}
    for item in popR:
        key = item[0]
        popD.setdefault(key, []).append(item[-1])
    return popD

#========================= MAIN ===================================================================
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()
    
    file_name = sys.argv[1]    
    c, items = read_problem(file_name)
    n = len(items)
    
    #===== GA parameters ========
    K = 2 # tournament size
    pX = 1 # Overall crossover rate
    pU = 0.5 # Uniform crossover rate
        
    if(n < 40):
        max_generations = 17
        P = 15000 # Population size
        E = 7500 # number of Elites
        Opt = 1
        pM = 40/100
        delHeuristic = 0.8 # prob of 0.2 accept heuristic for deletion
        random.seed(34)
    elif(n < 300):
        max_generations = 5
        P = 5000 # Population size
        E = 2500 # number of Elites
        Opt = 0.9
        pM = 1/100
        delHeuristic = 0.5
        random.seed(321)
    elif(n < 500):
        max_generations = 35
        P = 10000 # Population size
        E = 5000 # number of Elites
        Opt = 0.9
        pM = 1/100
        delHeuristic = 0.5
        random.seed(324)
    elif(n < 15000):
        max_generations = 50    
        P = 4000 # Population size
        E = 2000 # number of Elites
        Opt = 0.9
        pM = 1/100
        delHeuristic = 0.5
        random.seed(321)
    else:
        max_generations = 50    
        P = 4000 # Population size
        E = 2000 # number of Elites
        Opt = 0.9
        pM = 1/100
        delHeuristic = 0.5
        random.seed(321)
   
    # Create an initial population
    popL = initialize_population()
    popR = rankedList()
    popD = rankedDict()
    s = 0 # the generation counter
    bestResults = updateBest()
    
    while True:
        s += 1
        popR = rankedList()
        popD = rankedDict()
    
        # Update current best
        if max(popD) > bestResults[0]:
            bestResults = updateBest()
    
        # Stop if time is up
        if s == max_generations:
            break            
   
        # Start the child generation with E elites (direct copies from current generation)
        nextGen = select_elites()
            
        # Fill the next generation to size P, same size as the previous one
        while len(nextGen) < P:
            parentA = tournament_selection(popL, K) # Selection
            parentB = tournament_selection(popL, K)
            childrenAB = unifXover(parentA, parentB) # Crossover
            mutatedChildA = mutate(childrenAB[0]) # Mutation
            mutatedChildB = mutate(childrenAB[1])

            if(wFitness(mutatedChildA) > c):
                mutatedChildA = sackOptimizer(mutatedChildA, Opt, delHeuristic)
            if(wFitness(mutatedChildB) > c):
                mutatedChildB = sackOptimizer(mutatedChildB, Opt, delHeuristic)
            nextGen.extend([mutatedChildA, mutatedChildB])         
                   
        popL = nextGen[:]
    
    tot_value = bestResults[0]
    final_sequence = bestResults[1]
    
    print(tot_value)
    print('%d' % final_sequence[0], end='')
    for t in final_sequence[1:]:
        print(' %d' % t, end='')