#####################################     INFO     #######################################################
# Problems are represented as a 81-bit string, where each bit represents a cell on the grid. 0 represents 
# null space.

############################################################################################################

#IMPORTS
from random import choice, randint, shuffle, uniform, random
from collections import Counter
import numpy as np
import json
import os
import math
from tqdm import tqdm
import ast 
import copy
import tensorflow as tf
from math import sqrt
import copy

#CONSTANTS
GRID_SIZE = 9
POP_SIZE = 100
SELECTION_SIZE = 40
MUTATION_PROB = 0.4
MAX_GENERATIONS = 1000
ALLOW_DUPLICATES = False
NUMBER_OF_MUTATION_TYPES = 6

# This sets the path relative to the script's location
# Adjust the number of os.path.dirname() calls based on the script's depth relative to the 'Implementation' folder
script_directory = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(script_directory, 'DATA_2')
PATH_TO_GRIDS = os.path.join(script_directory, 'GRIDS')

# Now PATH_TO_DATA is set to '../Implementation/DATA/' relative to the script's location

'''
    [6, 2, 7, 8, 5, 1, 3, 4, 9],
    [5, 4, 3, 9, 2, 6, 8, 7, 1],
    [1, 9, 8, 4, 7, 3, 2, 5, 6],
    [2, 6, 1, 7, 3, 4, 5, 9, 8],
    [0, 7, 4, 6, 8, 5, 1, 3, 2],
    [8, 3, 5, 2, 1, 9, 7, 6, 4],
    [0, 1, 9, 5, 6, 8, 4, 2, 7],
    [7, 8, 6, 3, 4, 2, 9, 1, 5],
    [4, 5, 2, 1, 9, 7, 6, 8, 3]
'''
sample_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 9, 8, 4, 7, 3, 2, 5, 6],
    [2, 6, 1, 7, 3, 4, 5, 9, 8],
    [0, 7, 4, 6, 8, 5, 1, 3, 2],
    [8, 3, 5, 2, 1, 9, 7, 6, 4],
    [0, 1, 9, 5, 6, 8, 4, 2, 7],
    [7, 8, 6, 3, 4, 2, 9, 1, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]
'''
    [6, 0, 7, 0, 0, 1, 3, 4, 9],
    [5, 0, 3, 9, 2, 0, 8, 0, 1],
    [0, 9, 0, 4, 7, 3, 0, 5, 0],
    [2, 6, 0, 0, 3, 0, 5, 0, 0],
    [0, 0, 0, 0, 8, 0, 1, 0, 2],
    [8, 0, 5, 2, 0, 0, 7, 6, 0],
    [0, 0, 9, 5, 0, 8, 4, 0, 7],
    [7, 0, 0, 0, 4, 2, 9, 0, 5],
    [4, 5, 0, 1, 9, 0, 6, 0, 0]
'''
sample_grid2 = [
    [6, 0, 7, 0, 0, 1, 3, 4, 9],
    [5, 0, 3, 9, 2, 0, 8, 0, 1],
    [0, 9, 0, 4, 7, 3, 0, 5, 0],
    [2, 6, 0, 0, 3, 0, 5, 0, 0],
    [0, 0, 0, 0, 8, 0, 1, 0, 2],
    [8, 0, 5, 2, 0, 0, 7, 6, 0],
    [0, 0, 9, 5, 0, 8, 4, 0, 7],
    [7, 0, 0, 0, 4, 2, 9, 0, 5],
    [4, 5, 0, 1, 9, 0, 6, 0, 0]
]

sample_grid2 = [
    [0,5,0,0,0,6,0,4,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,6,0,0,0,0,0],
    [0,0,0,0,0,0,5,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,6,0,0,0],
    [0,0,0,0,5,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0]
    
]



class Grid ():
    #CONSTRAINED [i][j] = True if cell is immutable
    #                   = False if cell is mutable
    #create ref grid that shows where legal placement is
    def __init__(self, grid):
        self.grid = [row[:] for row in grid] 
        self.constrained_grid = [[True for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.fitness = -1
        self.prev_mutation = 0 # no mutation
        self.createGridConstraints()
        self.evalAllFitness()

    def createGridConstraints(self):
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                if self.grid[row][column] != 0:
                    self.constrained_grid[row][column] = False


    def clone(self):
        # Create a new grid array to ensure no references to the original grid
        cloned_grid_data = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

        # Fill the new grid with values from the original, respecting constraints
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                cloned_grid_data[i][j] = self.grid[i][j]

        # Instantiate a new Grid object with the cloned data
        cloned_grid = Grid(cloned_grid_data)
        cloned_grid.constrained_grid = [row[:] for row in self.constrained_grid]
        cloned_grid.fitness = self.fitness  # Copy fitness
        cloned_grid.prev_mutation = self.prev_mutation  # Copy any other relevant internal states

        return cloned_grid
    


#-----------------------------------------------SHOW---------------------------------------------------

    def showGrid(self):
        for row in self.grid:
            print(row)

    def showConstrainedGrid(self):
        for row in self.constrained_grid:
            print(row)

#-----------------------------------------------FITNESS------------------------------------------------
            
    #method to evaluate the fitness of a given solution - row based
    def evalFitnessRow(self, grid):
        fitness = 0
        for row in grid:
            #make dict of row
            rowCounter = Counter(row)
            for key, val in rowCounter.items():
                if val > 1:
                    fitness += (val - 1)

        return fitness


    #method to evaluate the fitness of a given solution - column based
    def evalFitnessColumn(self, grid):
        fitness = 0
        for column in range(len(grid[0])):
            rowCounter = Counter([grid[i][column] for i in range(len(grid))])    
            for key, val in rowCounter.items():
                if val > 1:
                    fitness += (val - 1)

        return fitness
    

    #method to evaluate the fitness of a given solution - subgrid based
    def evalFitnessSubGrid(self, grid):
        fitness = 0
        subGridSize = int(GRID_SIZE ** 0.5)
        for x in range(0, GRID_SIZE, subGridSize):
            for y in range(0, GRID_SIZE, subGridSize):
                subGrid = [grid[i][j] for i in range(x, x + subGridSize) for j in range(y, y + subGridSize)]
                counter = Counter(subGrid)
                for key, val in counter.items():
                    if val > 1:
                        fitness += (val - 1)
        return fitness
         

    #method to evaluate the fitness of a given solution - all fitnesses
    def evalAllFitness(self):
        self.fitness =  self.evalFitnessRow(self.grid) + self.evalFitnessColumn(self.grid) + self.evalFitnessSubGrid(self.grid)



#-----------------------------------------------EVOLUTIONARY ALGORITHM------------------------------------------------

class EvolutionaryAlgorithm ():
    def __init__(self, grid, guided_bool = False, model = None):
        self.population = []
        self.generation = 0
        self.iterations_since_best_fitness = 0
        self.best_fitness = 1000000000000
        self.initialise(grid)
        #variable methods for selection and mutation 
        self.mutation_selecter = None
        self.selection_method = None
        self.crossover_method = None
        self.guided_bool = guided_bool
        self.model = model
        self.class_rewards = {i: [] for i in range(NUMBER_OF_MUTATION_TYPES)}

        self.class_rewards_copy = None



    def initialise(self, grid):  
        self.GenPop(grid)


    def run_generation(self, max_generations):
        #selection
        parents = self.selection_method()
        #crossover
        for parent in range(0, len(parents), 2):
            offspring1, offspring2 = self.crossover_method(parents[parent].grid, parents[parent+1].grid, parents[parent].constrained_grid)
            #mutation according to mutator
            mutation_bound_1 = random()
            mutation_bound_2 = random()
            if mutation_bound_1 < MUTATION_PROB:
                self.mutation_selecter(offspring1)
            if mutation_bound_2 < MUTATION_PROB:
                self.mutation_selecter(offspring2)
            #add offspring to population
            self.population.append(offspring1)
            self.population.append(offspring2)
        #replacement
        self.Replacement()
        #check termination
        if self.population[0].fitness == 0 or self.generation == max_generations:
            return self.population[0].fitness
        #increase generation count
        if self.iterations_since_best_fitness > 100:
            map(self.mutationSelectionRandom, self.population[1:])
        #check if best fitness has improved
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.iterations_since_best_fitness = 0
        else:
            self.iterations_since_best_fitness += 1
        self.generation += 1




    def run_generation_2(self, max_generations):
        if self.generation == 0:
            [self.regenerateAll(candidate) for candidate in self.population[1:]]
        #selection
        parents = self.selection_method()
        #crossover
        for parent in range(0, len(parents), 2):
            offspring1, offspring2 = self.crossover_method(parents[parent].grid, parents[parent+1].grid, parents[parent].constrained_grid)
            #add offspring to population
            self.population.append(offspring1)
            self.population.append(offspring2)

        #mutation loop
        for candidate in range(len(self.population)):
            #ensure elitism
            if candidate == 0:
                #copy the candidate then mutate and add to population
                new_candidate_1, new_candidate_2 = self.crossoverAlternateRows(self.population[candidate].grid, self.population[candidate].grid, self.population[candidate].constrained_grid)
                #self.mutation_selecter(new_candidate_1)
                #self.mutation_selecter(new_candidate_2)
                self.normalise(new_candidate_1)
                self.normalise(new_candidate_2)
                if new_candidate_1.fitness < self.population[0].fitness or new_candidate_2.fitness < self.population[0].fitness:
                    input("a")
                self.population.append(new_candidate_1)
                self.population.append(new_candidate_2)
            else:
                #mutation according to mutator
                mutation_bound = random()
                if mutation_bound < MUTATION_PROB:
                    self.mutation_selecter(self.population[candidate])

        #replacement
        self.Replacement()
        #increase generation count
        #if self.iterations_since_best_fitness > 250:
        #if self.generation == 0:
        #    [self.regenerateAll(candidate) for candidate in self.population[1:]]
        #    print("Refreshing")
        #    self.iterations_since_best_fitness = 0
        #check if best fitness has improved
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.iterations_since_best_fitness = 0
        else:
            self.iterations_since_best_fitness += 1
        self.generation += 1




    def run_generation_guided_dep(self, max_generations):
        if self.generation == 0:
            [self.regenerateAll(candidate) for candidate in self.population[1:]]
        with tf.GradientTape() as tape:
            #initialise rewards dict
            class_rewards = {i: [] for i in range(NUMBER_OF_MUTATION_TYPES)}

            # Get previous diversity
            prev_diversity = self.diversity_function()

            # Selection of parents
            parents = self.selection_method()  # Ensure this returns actual parent objects
            new_offspring = []  # List to collect new offspring

            # Crossover and mutation
            for i in range(0, len(parents), 2):
                offspring1, offspring2 = self.crossover_method(parents[i].grid, parents[i+1].grid, parents[i].constrained_grid)
                for offspring in [offspring1, offspring2]:
                    if random() < MUTATION_PROB:
                        original_fitness = offspring.fitness
                        # Prepare inputs for the model
                        grid_tensor = np.array(offspring.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
                        scalar_inputs = np.array([offspring.fitness, offspring.prev_mutation, self.generation, self.iterations_since_best_fitness])
                        # Predict mutation action and apply
                        action_probs = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
                        action = np.argmax(action_probs[0])
                        self.mutationSelectionNN(offspring, action)
                        # Calculate reward
                        reward = self.reward_function(original_fitness, offspring.fitness, prev_diversity)
                        # Accumulate reward by action class
                        class_rewards[action].append(reward)
                    new_offspring.append(offspring)  # Add processed offspring to new offspring list
            
            # Calculate loss for each action class and sum up total loss
            total_loss = 0
            for action_class, rewards in class_rewards.items():
                if rewards:
                    # Aggregate rewards for this class
                    average_reward = sum(rewards) / len(rewards)
                    # Get class probabilities from the latest predictions
                    class_prob = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])[0, action_class]
                    # Calculate loss for this class
                    loss = -tf.math.log(class_prob + 1e-7) * average_reward  # Add epsilon to avoid log(0)
                    total_loss += loss

            #replacement
            self.population += new_offspring
            self.Replacement()
            # Apply gradients to update the model
            gradients = tape.gradient(total_loss, self.model.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.model.trainable_variables))

        '''#increase generation count
        if self.iterations_since_best_fitness > 100:
            map(self.regenerateAll, self.population[1:])
            print("REFRESHING")'''
        #check if best fitness has improved
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.iterations_since_best_fitness = 0
        else:
            self.iterations_since_best_fitness += 1
        self.generation += 1


        # Check termination condition
        if self.population[0].fitness == 0 or self.generation >= max_generations:
            return True  # Termination condition met




    def run_generation_guided_dep_2(self, max_generations, update_interval=1):
        if self.generation == 0:
            [self.regenerateAll(candidate) for candidate in self.population[1:]]
        if self.generation % update_interval == 0:
            with tf.GradientTape() as tape:
                # Get previous diversity
                prev_diversity = self.diversity_function()
                best_fitness = self.population[0].fitness

                # Selection of parents
                parents = self.selection_method()  # Ensure this returns actual parent objects
                new_offspring = []  # List to collect new offspring

                # Crossover and mutation
                for i in range(0, len(parents), 2):
                    offspring1, offspring2 = self.crossover_method(parents[i].grid, parents[i+1].grid, parents[i].constrained_grid)
                    for offspring in [offspring1, offspring2]:
                        if random() < MUTATION_PROB:
                            original_fitness = offspring.fitness
                            # Prepare inputs for the model
                            grid_tensor = np.array(offspring.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
                            scalar_inputs = np.array([offspring.fitness, offspring.prev_mutation, self.generation, self.iterations_since_best_fitness])
                            # Predict mutation action and apply
                            action_probs = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
                            action = np.argmax(action_probs[0])
                            self.mutationSelectionNN(offspring, action)
                            # Calculate reward
                            reward = self.reward_function(original_fitness, offspring.fitness, prev_diversity, best_fitness)
                            # Accumulate reward by action class
                            self.class_rewards[action].append(reward)
                        new_offspring.append(offspring)  # Add processed offspring to new offspring list
                
                # Calculate loss for each action class and sum up total loss
                total_loss = 0
                for action_class, rewards in self.class_rewards.items():
                    if rewards:
                        # Aggregate rewards for this class
                        average_reward = sum(rewards) / len(rewards)
                        # Get class probabilities from the latest predictions
                        class_prob = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])[0, action_class]
                        # Calculate loss for this class
                        loss = -tf.math.log(class_prob + 1e-7) * average_reward  # Add epsilon to avoid log(0)
                        total_loss += loss

                #replacement
                self.population += new_offspring
                self.Replacement()
                # Apply gradients to update the model
                gradients = tape.gradient(total_loss, self.model.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.model.trainable_variables))
                #reset rewards
                self.class_rewards_copy = copy.deepcopy(self.class_rewards)
                self.class_rewards = {i: [] for i in range(NUMBER_OF_MUTATION_TYPES)}
        else:
            # Get previous diversity
            prev_diversity = self.diversity_function()
            best_fitness = self.population[0].fitness

            # Selection of parents
            parents = self.selection_method()  # Ensure this returns actual parent objects
            new_offspring = []  # List to collect new offspring

            # Crossover and mutation
            for i in range(0, len(parents), 2):
                offspring1, offspring2 = self.crossover_method(parents[i].grid, parents[i+1].grid, parents[i].constrained_grid)
                for offspring in [offspring1, offspring2]:
                    if random() < MUTATION_PROB:
                        original_fitness = offspring.fitness
                        # Prepare inputs for the model
                        grid_tensor = np.array(offspring.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
                        scalar_inputs = np.array([offspring.fitness, offspring.prev_mutation, self.generation, self.iterations_since_best_fitness])
                        # Predict mutation action and apply
                        action_probs = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
                        action = np.argmax(action_probs[0])
                        self.mutationSelectionNN(offspring, action)
                        # Calculate reward
                        reward = self.reward_function(original_fitness, offspring.fitness, prev_diversity, best_fitness)
                        # Accumulate reward by action class
                        self.class_rewards[action].append(reward)
                    new_offspring.append(offspring)  # Add processed offspring to new offspring list
            
            # No model update this round, just add new offspring to the population
            self.population += new_offspring
            self.Replacement()

        '''#increase generation count
        if self.iterations_since_best_fitness > 100:
            map(self.regenerateAll, self.population[1:])
            print("REFRESHING")'''
        #check if best fitness has improved
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.iterations_since_best_fitness = 0
        else:
            self.iterations_since_best_fitness += 1
        self.generation += 1


    #incorps ES strategies
    def run_generation_guided(self, max_generations, update_interval=1):
        if self.generation == 0:
            [self.regenerateAll(candidate) for candidate in self.population[1:]]
        if self.generation % update_interval == 0:
            with tf.GradientTape() as tape:
                # Get previous diversity
                prev_diversity = self.diversity_function()
                best_fitness = self.population[0].fitness

                # Selection of parents
                parents = self.selection_method()  # Ensure this returns actual parent objects
                new_offspring = []  # List to collect new offspring

                # Crossover 
                for i in range(0, len(parents), 2):
                    offspring1, offspring2 = self.crossover_method(parents[i].grid, parents[i+1].grid, parents[i].constrained_grid)
                    for offspring in [offspring1, offspring2]:
                        new_offspring.append(offspring)  # Add processed offspring to new offspring list
                
                #mutation
                self.population += new_offspring
                #apply ES mutate all except the best
                for offspring in self.population[1:]:
                    if random() < MUTATION_PROB:
                        original_fitness = offspring.fitness
                        # Prepare inputs for the model
                        grid_tensor = np.array(offspring.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
                        scalar_inputs = np.array([offspring.fitness, offspring.prev_mutation, self.generation, self.iterations_since_best_fitness])
                        # Predict mutation action and apply
                        action_probs = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
                        #action = np.argmax(action_probs[0])
                        # Normalize the probabilities
                        probabilities = action_probs.numpy().flatten()
                        probabilities /= np.sum(probabilities)
                        action = np.random.choice(len(probabilities), p=probabilities)
                        self.mutationSelectionNN(offspring, action)
                        # Calculate reward
                        reward = self.reward_function(original_fitness, offspring.fitness, prev_diversity, best_fitness)
                        # Accumulate reward by action class
                        self.class_rewards[action].append(reward)

                #print(f"self.class_rewards: {self.class_rewards}")
                self.Replacement()

                #get new diversity term
                new_diversity = self.diversity_function()
                
                div_term = (self.iterations_since_best_fitness - 1) * (new_diversity - prev_diversity)
                #print(f"div_term is {div_term}")
                # add diversity terms to the rewards
                for clabel, rewards in self.class_rewards.items():
                    if rewards:
                        for i in range(len(rewards)):
                            rewards[i] += div_term



                
                # Calculate loss for each action class and sum up total loss
                total_loss = 0
                total_classes = 0
                for action_class, rewards in self.class_rewards.items():
                    if rewards:
                        total_classes += 1
                        # Aggregate rewards for this class
                        average_reward = sum(rewards) / len(rewards)
                        #print(f"average_reward is {average_reward}")
                        # Get class probabilities from the latest predictions
                        class_prob = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])[0, action_class]
                        # Calculate loss for this class
                        loss = -tf.math.log(class_prob + 0.1 + 1e-7) * average_reward  # Add epsilon to avoid log(0)
                        total_loss += loss
                        
                if total_classes == 1:
                    total_loss += self.iterations_since_best_fitness

                if self.generation > 2:
                    print(f"total_loss is {total_loss}")    
                    # Apply gradients to update the model
                    gradients = tape.gradient(total_loss, self.model.model.trainable_variables)
                    #print(f"gradients are {gradients}")
                    clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                    self.model.optimizer.apply_gradients(zip(clipped_gradients, self.model.model.trainable_variables))
                    #reset rewards
                    self.class_rewards_copy = copy.deepcopy(self.class_rewards)
                    self.class_rewards = {i: [] for i in range(NUMBER_OF_MUTATION_TYPES)}
        else:
            # Get previous diversity
            prev_diversity = self.diversity_function()
            best_fitness = self.population[0].fitness

            # Selection of parents
            parents = self.selection_method()  # Ensure this returns actual parent objects
            new_offspring = []  # List to collect new offspring

            # Crossover and mutation
            for i in range(0, len(parents), 2):
                offspring1, offspring2 = self.crossover_method(parents[i].grid, parents[i+1].grid, parents[i].constrained_grid)
                for offspring in [offspring1, offspring2]:
                    new_offspring.append(offspring)  # Add processed offspring to new offspring list

            self.population += new_offspring
            for offspring in self.population[1:]:
                if random() < MUTATION_PROB:
                    original_fitness = offspring.fitness
                    # Prepare inputs for the model
                    grid_tensor = np.array(offspring.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
                    scalar_inputs = np.array([offspring.fitness, offspring.prev_mutation, self.generation, self.iterations_since_best_fitness])
                    # Predict mutation action and apply
                    action_probs = self.model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
                    action = np.argmax(action_probs[0])
                    self.mutationSelectionNN(offspring, action)
                    # Calculate reward
                    reward = self.reward_function(original_fitness, offspring.fitness, prev_diversity, best_fitness)
                    # Accumulate reward by action class
                    self.class_rewards[action].append(reward)


            self.Replacement()

            #update diversities
            #get new diversity term
            new_diversity = self.diversity_function()
            # add diversity terms to the rewards
            for clabel, rewards in self.class_rewards.items():
                if rewards:
                    for reward in rewards:
                        reward += (self.iterations_since_best_fitness - 1 if self.iterations_since_best_fitness > 1 else 0) * (new_diversity - (prev_diversity*1.25))



        '''#increase generation count
        if self.iterations_since_best_fitness > 100:
            map(self.regenerateAll, self.population[1:])
            print("REFRESHING")'''
        #check if best fitness has improved
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.iterations_since_best_fitness = 0
        else:
            self.iterations_since_best_fitness += 1
        self.generation += 1





        # Check termination condition
        if self.population[0].fitness == 0 or self.generation >= max_generations:
            return True  # Termination condition met



    def run(self):
        parents = []
        running = True
        best_fitness = 1000000000000
        allRunData = []
        while running == True:
            print(f"Generation = {self.generation}, best solution has fitness = {self.population[0].fitness}")
            #select the best candidates and generate new generation
            parents = self.selectionTournament()
            #generate offspring
            for parent in range(0, len(parents), 2):
                offspring1, offspring2 = self.crossoverCyclic(parents[parent].grid, parents[parent+1].grid, parents[parent].constrained_grid)
                #randomly mutate some offspring
                mutation_bound_1 = random()
                mutation_bound_2 = random()
                if mutation_bound_1 < MUTATION_PROB:
                    self.mutationSelectionRandom(offspring1)
                if mutation_bound_2 < MUTATION_PROB:
                    self.mutationSelectionRandom(offspring2)
                #add offspring to population
                self.population.append(offspring1)
                self.population.append(offspring2)
            #examine population and prune
            #replacement - use elitism
            self.Replacement()
            #check if best fitness has improved
            if self.population[0].fitness < best_fitness:
                best_fitness = self.population[0].fitness
                self.iterations_since_best_fitness = 0
            else:
                self.iterations_since_best_fitness += 1

            #continue until termination is met.
            if self.population[0].fitness == 0 or self.generation == MAX_GENERATIONS:
                print("TERMINIATED")
                return self

            self.generation += 1



    def runWithDataAcquistion(self):
        parents = []
        running = True
        best_fitness = 1000000000000
        #stores all data for each run, each generation
        #in the form of [grid, fitness, generation, iterations_since_best_fitness]
        all_run_data = []

        while running == True:
            #print(f"Generation = {self.generation}, best solution has fitness = {self.population[0].fitness}")
            #select the best candidates and generate new generation
            parents = self.selectionTournament()
            #generate offspring
            for parent in range(0, len(parents), 2):
                offspring1, offspring2 = self.crossoverCyclic(parents[parent].grid, parents[parent+1].grid, parents[parent].constrained_grid)
                #randomly mutate some offspring
                mutation_bound_1 = random()
                mutation_bound_2 = random()
                if mutation_bound_1 < MUTATION_PROB:
                    self.mutationSelectionRandom(offspring1)
                if mutation_bound_2 < MUTATION_PROB:
                    self.mutationSelectionRandom(offspring2)
                #add offspring to population
                self.population.append(offspring1)
                self.population.append(offspring2)
            #examine population and prune
            #replacement - use elitism
            self.Replacement()
            #check if best fitness has improved
            if self.population[0].fitness < best_fitness:
                best_fitness = self.population[0].fitness
                self.iterations_since_best_fitness = 0
            else:
                self.iterations_since_best_fitness += 1

            #add all information from this generation to all_run_data
            for candidate in self.population:
                all_run_data.append([candidate.grid, candidate.fitness, self.generation, self.iterations_since_best_fitness])

            #continue until termination is met.
            if self.population[0].fitness == 0 or self.generation == MAX_GENERATIONS:
                print("TERMINIATED")
                #returns all data, and the constraints
                return all_run_data, self.population[0].constrained_grid

            self.generation += 1



    def runWithDataAcquistionTraining(self):
        parents = []
        running = True
        best_fitness = 1000000000000
        #stores all data for each run, each generation
        #in the form of [grid, fitness, generation, iterations_since_best_fitness]
        all_run_data = []

        while running == True:
            #print(f"Generation = {self.generation}, best solution has fitness = {self.population[0].fitness}")
            # generate copy of only parent
            new_candidate = copy.copy(self.population[0])
            # mutate copy
            self.mutationSelectionRandom(new_candidate)
            # add to population
            self.population.append(new_candidate)
            #select the best candidates and generate new generation
            self.Replacement()
            #check if best fitness has improved
            if self.population[0].fitness < best_fitness:
                best_fitness = self.population[0].fitness
                self.iterations_since_best_fitness = 0
            else:
                self.iterations_since_best_fitness += 1

            #add all information from this generation to all_run_data
            for candidate in self.population:
                all_run_data.append([candidate.grid, candidate.fitness, candidate.prev_mutation, self.generation, self.iterations_since_best_fitness])

            #continue until termination is met.
            if self.population[0].fitness == 0 or self.generation == MAX_GENERATIONS:
                print("TERMINIATED")
                #returns all data, and the constraints
                return all_run_data, self.population[0].constrained_grid

            self.generation += 1




    #fills empty values according to content of row
    def generateCandidate(self, g = sample_grid2):
        grid = Grid(g)
        constrained_g = grid.constrained_grid
        for row in range(len(constrained_g)):
            #find values missing from row
            missing_in_row = list(set([i for i in range(1,10)]) - set(grid.grid[row]))
            #introduce extra randomness
            shuffle(missing_in_row)
            #find blank spaces, input values missing from rows
            for column in range(len(constrained_g[row])): 
                if constrained_g[row][column] == True:
                    grid.grid[row][column] = missing_in_row.pop(randint(0,len(missing_in_row)-1))
        grid.evalAllFitness()
        return grid


    #Generate initial pop
    def GenPop(self, g = sample_grid):
        for i in range(POP_SIZE):
            self.population.append(self.generateCandidate(g))



    def Replacement(self):
        #preserve the best candidates
        new_population = []
        #sort
        self.population.sort(key=lambda grid: grid.fitness)
        #preserve best 10% and worst 10%
        new_population += self.population[:int(POP_SIZE/20)]
        new_population += self.population[-int(POP_SIZE/20):]

        #remove best and worst from random sample set
        other_cands = self.population[int(POP_SIZE/20):-int(POP_SIZE/20)]

        for i in range(18*(int(POP_SIZE/20))):
            new_population.append(choice(other_cands))

        self.population = new_population



    #---------------------------------------NN STUFF---------------------------------------------------
    #combination of fitness metric and diversirt metric
    def reward_function(self, prev_fitness, current_fitness, prev_diversity, best_fitness):
        fitness_term = self.fitness_reward(prev_fitness, current_fitness)
        if current_fitness < best_fitness and current_fitness > prev_fitness:
            fitness_term *= (1/(best_fitness - current_fitness)**3)
        reward = fitness_term #+ (self.iterations_since_best_fitness - 1) * (diversity_term - prev_diversity)
        return reward

    #considers difference in fitness, and the number of iterations since best fitness, -0.5 term there 
    #to immeadiately punish stagnation
    def fitness_reward(self, prev_fitness, current_fitness):
        return self.iterations_since_best_fitness*(current_fitness-prev_fitness-0.5)


    #uses entropy to measure the diversity of the population
    def diversity_function(self):
        # Convert continuous fitness values to discrete categories if necessary
        fitness_values = [individual.fitness for individual in self.population]
        # Assuming fitness values are discrete or have been discretized:
        values, counts = np.unique(fitness_values, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small number to avoid log(0)
        return entropy


    #---------------------------------------SELECTION---------------------------------------------------
    def selectionRouletteWheel(self):
        # First, find the maximum fitness to invert the selection criteria (lower is better)
        max_fitness = max(grid.fitness for grid in self.population if grid.fitness > 0) + 1

        # Calculate selection probabilities inversely proportional to fitness
        inverted_fitness_totals = sum(max_fitness - grid.fitness for grid in self.population)
        selection_probs = [(max_fitness - grid.fitness) / inverted_fitness_totals for grid in self.population]

        selected = []
        while len(selected) < SELECTION_SIZE:
            pick = uniform(0, 1)
            current = 0
            for grid, prob in zip(self.population, selection_probs):
                current += prob
                if current > pick:
                    if ALLOW_DUPLICATES == True or (grid not in selected):
                        selected.append(grid)
                        break

        return selected

    

    def selectionTournament(self, tournament_size=5):
        selected = []
        while len(selected) < SELECTION_SIZE:
            # Select candidates for the tournament
            tournament_candidates = [choice(self.population) for _ in range(tournament_size)]
            # Find the candidate with the lowest fitness
            best_candidate = min(tournament_candidates, key=lambda grid: grid.fitness)

            # Check for duplicates if they are not allowed
            if ALLOW_DUPLICATES or (best_candidate not in selected):
                selected.append(best_candidate)
        return selected


    

    def selectionRankBased(self):
        # Sort the population based on fitness, lower is better
        ranked_population = sorted(self.population, key=lambda grid: grid.fitness)

        # Assign selection probabilities based on inverse rank (linearly, in this case)
        total_ranks = sum(range(1, len(self.population) + 1))
        selection_probs = [rank / total_ranks for rank in range(len(self.population), 0, -1)]

        selected = []
        while len(selected) < SELECTION_SIZE:
            pick = uniform(0, 1)
            current = 0
            for grid, prob in zip(ranked_population, selection_probs):
                current += prob
                if current > pick:
                    if ALLOW_DUPLICATES or (grid not in selected):
                        selected.append(grid)
                        break

        return selected
        
    #---------------------------------------CROSSOVER---------------------------------------------------
    
    def crossoverAlternateRows(self, parent1, parent2, constrained_grid):
        offspring = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        offspring2 = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        which_parent = 1
        for i in range(len(offspring)):
            for j in range(len(offspring)):
                #check if cell is immutable
                if constrained_grid[i][j] == False:
                    #which parents turn 
                    if which_parent == 1:
                        offspring[i][j] = parent1[i][j]
                        offspring2[i][j] = parent2[i][j]
                        which_parent *= -1
                    else:
                        offspring[i][j] = parent2[i][j]
                        offspring2[i][j] = parent1[i][j]
                        which_parent *= -1
                else:
                    offspring[i][j] = parent1[i][j]
                    offspring2[i][j] = parent1[i][j]
        
        offspring = Grid(offspring)
        offspring2 = Grid(offspring2)
        offspring.constrained_grid = constrained_grid
        offspring2.constrained_grid = constrained_grid
        offspring.evalAllFitness()
        offspring2.evalAllFitness()
        return offspring, offspring2
        


    def crossoverCyclic(self, parent1, parent2, constrained_grid):
        def findCycle(start, row1, row2):
            try:
                cycle = set()
                current = start
                while True:
                    cycle.add(current)
                    # Find the value in row1 and its position in row2
                    value = row1[current]
                    current = row2.index(value)
                    if current in cycle:
                        break
                return cycle
            except ValueError:
                return []

        def cyclicCrossoverRow(row1, row2, constrainedRow):
            offspringRow = [0] * len(row1)
            start = 0  # Starting from the first cell of the row
            cycle = findCycle(start, row1, row2)

            for i in range(len(row1)):
                if constrainedRow[i] == True:  # Check if cell is mutable
                    offspringRow[i] = row1[i] if i in cycle else row2[i]
                else:
                    offspringRow[i] = row1[i]  # Preserve constrained value

            return offspringRow

        offspring1 = []
        offspring2 = []
        for i in range(len(parent1)):
            rowOffspring1 = cyclicCrossoverRow(parent1[i], parent2[i], constrained_grid[i])
            rowOffspring2 = cyclicCrossoverRow(parent2[i], parent1[i], constrained_grid[i])  # Reverse roles of parents
            offspring1.append(rowOffspring1)
            offspring2.append(rowOffspring2)
    
        offspring1 = Grid(offspring1)
        offspring2 = Grid(offspring2)
        offspring1.constrained_grid = constrained_grid
        offspring2.constrained_grid = constrained_grid
        offspring1.evalAllFitness()
        offspring2.evalAllFitness()

        return offspring1, offspring2



    def crossoverPMX(self, parent1, parent2, constrained_grid):
        def createMapping(subsection1, subsection2):
            return {subsection1[i]: subsection2[i] for i in range(len(subsection1))}

        def applyMapping(row, mapping):
            return [mapping.get(element, element) for element in row]

        def partiallyMappedCrossoverRow(row1, row2, constrainedRow):
            start, end = sorted([randint(0, len(row1) - 1) for _ in range(2)])
            subsection1 = row1[start:end+1]
            subsection2 = row2[start:end+1]

            mapping1 = createMapping(subsection1, subsection2)
            mapping2 = createMapping(subsection2, subsection1)

            offspringRow1 = row1[:]
            offspringRow2 = row2[:]

            # Apply mapping outside the crossover section
            for i in range(len(row1)):
                if not start <= i <= end:
                    if constrainedRow[i] == True:  # If cell is mutable
                        offspringRow1[i] = applyMapping([row2[i]], mapping2)[0]
                        offspringRow2[i] = applyMapping([row1[i]], mapping1)[0]

            return offspringRow1, offspringRow2

        offspring1 = []
        offspring2 = []
        for i in range(len(parent1)):
            rowOffspring1, rowOffspring2 = partiallyMappedCrossoverRow(parent1[i], parent2[i], constrained_grid[i])
            offspring1.append(rowOffspring1)
            offspring2.append(rowOffspring2)
        
        offspring1 = Grid(offspring1)
        offspring2 = Grid(offspring2)
        offspring1.constrained_grid = constrained_grid
        offspring2.constrained_grid = constrained_grid
        offspring1.evalAllFitness()
        offspring2.evalAllFitness()

        return offspring1, offspring2



    #---------------------------------------MUTATIONS---------------------------------------------------


    def swap(self, candidate):
        #select two random locations in the grid, that are not immutable
        positions_valid = False
        while positions_valid == False:
            pos1 = (randint(0, GRID_SIZE-1), randint(0, GRID_SIZE-1))
            pos2 = (randint(0, GRID_SIZE-1), randint(0, GRID_SIZE-1))
            if candidate.constrained_grid[pos1[0]][pos1[1]] == False and candidate.constrained_grid[pos2[0]][pos2[1]] == False and pos1 != pos2:
                positions_valid = True
        #swap the values at these locations
        candidate.grid[pos1[0]][pos1[1]], candidate.grid[pos2[0]][pos2[1]] = candidate.grid[pos2[0]][pos2[1]], candidate.grid[pos1[0]][pos1[1]]



    def flip(self, candidate):
        new_row = [_ for _ in range(GRID_SIZE)]
        #select row, column or grid as target
        choice = randint(0, 2)
        #randomly select a one of the above, out of GRID_SIZE
        target = randint(0, GRID_SIZE-1)
        #flip the values in the target, has to consider grid constraints
        #row
        if choice == 0:
            flip_content = candidate.grid[target]
            for i in range(len(flip_content)):
                #flip the value if the cell is mutable
                if candidate.constrained_grid[target][i] == False:
                    new_row[i] = flip_content[GRID_SIZE-1-i]
                else:
                    new_row[i] = flip_content[i]
            #update the row
            candidate.grid[target] = new_row


        # column
        elif choice == 1:
            flip_content = np.array(candidate.grid)[:, target]
            for i in range(len(flip_content)):
                if candidate.constrained_grid[i][target] == False:
                    new_row[i] = flip_content[GRID_SIZE-1-i]
                else:
                    new_row[i] = flip_content[i]
            #update the column
            for i in range(len(new_row)):
                candidate.grid[i][target] = new_row[i]

        # grid
        elif choice == 2:
            subGridSize = int(GRID_SIZE ** 0.5)
            x = (target // subGridSize) * subGridSize
            y = (target % subGridSize) * subGridSize
            flip_content = np.array(candidate.grid)[x:x+subGridSize, y:y+subGridSize].flatten()
            flip_content = flip_content.reshape((subGridSize, subGridSize))
            # Initialize new_row here
            new_row = [[0 for _ in range(subGridSize)] for _ in range(subGridSize)]

            for i in range(len(flip_content)):
                for j in range(len(flip_content[i])):
                    if candidate.constrained_grid[x+i][y+j] == False:
                        new_row[i][j] = flip_content[int(GRID_SIZE**0.5)-1-i][int(GRID_SIZE**0.5)-1-j]
                    else:
                        new_row[i][j] = flip_content[i][j]

            # Update the grid
            for i in range(len(new_row)):
                for j in range(len(new_row[i])):
                    candidate.grid[x+i][y+j] = new_row[i][j]

        else:
            print("error, invalid choice for flip mutation")
        #update fitness
        candidate.evalAllFitness()    



    def regenerate(self, candidate):
        #select start and end of the target
        start = randint(0, GRID_SIZE**2)
        end = randint(0, GRID_SIZE**2)
        current = 0
        if end < start:
            end, start = start, end
        #iterate through the target, and regenerate the values, obeying grid constraints
        while True:
            if current > GRID_SIZE**2-1:
                break
            elif current < start or current > end:
                pass
            elif current >= start and current <= end:
                #check if the cell is mutable
                if candidate.constrained_grid[current//GRID_SIZE][current%GRID_SIZE] == False:
                    #regenerate
                    candidate.grid[current//GRID_SIZE][current%GRID_SIZE] = randint(1,GRID_SIZE)
            current += 1
        #update fitness
        candidate.evalAllFitness()

        

    def smartRegenerate(self, candidate):
        start = randint(0, GRID_SIZE**2)
        end = randint(0, GRID_SIZE**2)
        if end < start:
            end, start = start, end

        for current in range(GRID_SIZE**2):
            if start <= current <= end:
                x, y = current // GRID_SIZE, current % GRID_SIZE
                if not candidate.constrained_grid[x][y]:
                    # Get existing values in the row
                    existing_values = set(candidate.grid[x])
                    # Generate a list of possible new values
                    possible_values = [v for v in range(1, GRID_SIZE+1) if v not in existing_values]
                    # Check if there are any possible values to choose from
                    if possible_values:
                        new_value = choice(possible_values)
                        candidate.grid[x][y] = new_value
                    else:
                        # Handle the case where no new value can be chosen
                        # This could be skipping the cell, or some other logic
                        pass

        candidate.evalAllFitness()


    def normalise(self, candidate):
        GRID_SIZE = len(candidate.grid)  # Assuming this is defined somewhere globally if not passed
        changed = False
        sub_grid_size = int(math.sqrt(GRID_SIZE))  # Calculate size of each subgrid

        # Helper function to get available numbers for a cell considering row, column, and subgrid
        def get_available_numbers(x, y):
            row_values = set(candidate.grid[x])
            column_values = set(candidate.grid[i][y] for i in range(GRID_SIZE))
            block_start_x, block_start_y = (x // sub_grid_size) * sub_grid_size, (y // sub_grid_size) * sub_grid_size
            subgrid_values = set(candidate.grid[i][j] for i in range(block_start_x, block_start_x + sub_grid_size) for j in range(block_start_y, block_start_y + sub_grid_size))
            all_used = row_values.union(column_values).union(subgrid_values)
            return set(range(1, GRID_SIZE + 1)) - all_used

        # Normalize each component: rows, columns, and subgrids
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if candidate.constrained_grid[x][y] == False:  # If cell is mutable
                    available_numbers = get_available_numbers(x, y)
                    current_value = candidate.grid[x][y]
                    if current_value in available_numbers:
                        available_numbers.remove(current_value)  # Avoid setting the same number
                    if available_numbers:
                        candidate.grid[x][y] = available_numbers.pop()
                        candidate.evalAllFitness()
                        changed = True
                        print("changed")

        return changed  # Optionally return whether a change was made


    def normalise2(self, candidate):
        changed = False  # Flag to track if any changes have been made in the rows
        # Normalize rows
        for x in range(GRID_SIZE):
            row = candidate.grid[x]
            row_set = set(row)
            if len(row_set) < GRID_SIZE:  # Check for duplicate values in the row
                for value in row_set:
                    if row.count(value) > 1:  # Find the duplicate value
                        # Find indices of duplicate value
                        duplicate_indices = [i for i, v in enumerate(row) if v == value]
                        for index in duplicate_indices:
                            if candidate.constrained_grid[x][index] == False:  # Check if cell is mutable
                                # Find a new value for this cell
                                new_value = next((v for v in range(1, GRID_SIZE+1) if v not in row_set), None)
                                if new_value:
                                    candidate.grid[x][index] = new_value
                                    candidate.evalAllFitness()
                                    changed = True
                                if changed:
                                    break
                        if changed:
                            break
                if changed:
                    break

        # If no changes were made in rows, normalize columns
        if not changed:
            for y in range(GRID_SIZE):
                column = [candidate.grid[x][y] for x in range(GRID_SIZE)]
                col_set = set(column)
                if len(col_set) < GRID_SIZE:
                    for value in col_set:
                        if column.count(value) > 1:
                            duplicate_indices = [x for x, v in enumerate(column) if v == value]
                            for index in duplicate_indices:
                                if candidate.constrained_grid[index][y] == False:
                                    new_value = next((v for v in range(1, GRID_SIZE+1) if v not in col_set), None)
                                    if new_value:
                                        candidate.grid[index][y] = new_value
                                        candidate.evalAllFitness()
                                        changed = True
                                    if changed:
                                        break
                            if changed:
                                break
                    if changed:
                        break
            
        # Normalize subgrids if no changes in rows and columns
        if not changed:
            for block_start_x in range(0, GRID_SIZE, int(sqrt(GRID_SIZE))):
                for block_start_y in range(0, GRID_SIZE, int(sqrt(GRID_SIZE))):
                    subgrid = [candidate.grid[i][j] for i in range(block_start_x, block_start_x + int(sqrt(GRID_SIZE))) for j in range(block_start_y, block_start_y + int(sqrt(GRID_SIZE)))]
                    sub_set = set(subgrid)
                    if len(sub_set) < GRID_SIZE:
                        for value in sub_set:
                            if subgrid.count(value) > 1:
                                duplicate_indices = [(i, j) for i in range(block_start_x, block_start_x + int(sqrt(GRID_SIZE))) for j in range(block_start_y, block_start_y + int(sqrt(GRID_SIZE))) if candidate.grid[i][j] == value]
                                for index_x, index_y in duplicate_indices:
                                    if candidate.constrained_grid[index_x][index_y] == False:
                                        new_value = next((v for v in range(1, GRID_SIZE+1) if v not in sub_set), None)
                                        if new_value:
                                            candidate.grid[index_x][index_y] = new_value
                                            candidate.evalAllFitness()
                                            changed = True
                                        if changed:
                                            break
                                if changed:
                                    break
                        if changed:
                            break

        return changed  # Optionally return whether a change was made
#


    def FilteredMutation(self, candidate):
        GRID_SIZE = len(candidate.grid)
        sub_grid_size = int(sqrt(GRID_SIZE))
        sub_grid_improved = False
        attempts = 0
        max_attempts = GRID_SIZE * GRID_SIZE  # Set a limit to prevent infinite loops

        while not sub_grid_improved and attempts < max_attempts:
            attempts += 1
            sub_grid_row = randint(0, sub_grid_size - 1) * sub_grid_size
            sub_grid_col = randint(0, sub_grid_size - 1) * sub_grid_size
            # Get positions in the sub-grid and shuffle them to ensure randomness
            positions = [(i, j) for i in range(sub_grid_row, sub_grid_row + sub_grid_size)
                        for j in range(sub_grid_col, sub_grid_col + sub_grid_size)
                        if not candidate.constrained_grid[i][j]]
            shuffle(positions)

            for idx, pos1 in enumerate(positions):
                for pos2 in positions[idx+1:]:
                    # Swap cells
                    candidate.grid[pos1[0]][pos1[1]], candidate.grid[pos2[0]][pos2[1]] = candidate.grid[pos2[0]][pos2[1]], candidate.grid[pos1[0]][pos1[1]]
                    candidate.evalAllFitness()
                    new_fitness = candidate.fitness
                    # Check if fitness improved, if not revert the swap
                    if new_fitness < candidate.fitness:
                        candidate.fitness = new_fitness
                        sub_grid_improved = True
                        break
                    else:
                        # Revert swap if no improvement
                        candidate.grid[pos1[0]][pos1[1]], candidate.grid[pos2[0]][pos2[1]] = candidate.grid[pos2[0]][pos2[1]], candidate.grid[pos1[0]][pos1[1]]
                if sub_grid_improved:
                    break

        #if not sub_grid_improved:
            #print("No feasible mutation found within the allowed attempts.")
        return sub_grid_improved
    

    def regenerateAll2(self, candidate):
        # Iterate through each cell in the grid
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Check if the cell is mutable
                if candidate.constrained_grid[row][col] == False:
                    # Regenerate the value for the mutable cell
                    candidate.grid[row][col] = randint(1, GRID_SIZE)
        # Update fitness after regeneration
        candidate.evalAllFitness()


    def regenerateAll(self, candidate):
        for row in range(GRID_SIZE):
            # Determine the available numbers for the row, considering immutability
            available_numbers = set(range(1, GRID_SIZE + 1)) - set(candidate.grid[row][col] for col in range(GRID_SIZE) if candidate.constrained_grid[row][col])
            available_numbers = list(available_numbers)
            shuffle(available_numbers)  # Shuffle the available numbers to ensure randomness in selection

            mutable_indices = [col for col in range(GRID_SIZE) if not candidate.constrained_grid[row][col]]
            shuffle(mutable_indices)  # Shuffle the mutable indices to distribute values randomly

            # Regenerate values for mutable cells without repeating any values on the row
            for index in mutable_indices:
                if available_numbers:  # Ensure there are still available numbers to use
                    candidate.grid[row][index] = available_numbers.pop()  # Assign and remove from available numbers

        # Update fitness after regeneration
        candidate.evalAllFitness()

#---------------------------------------MUTATION SELECTION---------------------------------------------------

    def mutationSelectionRandom(self, candidate):
        choice = randint(1, 6)
        if choice == 1:
            mutated_cand = self.swap(candidate)
        elif choice == 2:
            mutated_cand = self.flip(candidate)
        elif choice == 3:
            mutated_cand = self.regenerate(candidate)
        elif choice == 4:
            mutated_cand = self.smartRegenerate(candidate)
        elif choice == 5:
            mutated_cand = self.normalise(candidate)
        elif choice == 6:
            mutated_cand = self.FilteredMutation(candidate)
        candidate.prev_mutation = choice
        return mutated_cand


    def mutationSelectionRandom2(self, candidate):
        choice = randint(0, 2)
        if choice == 0:
            mutated_cand = self.normalise(candidate)
        elif choice == 1:
            mutated_cand = self.smartRegenerate(candidate)
        elif choice == 2:
            mutated_cand = self.regenerateAll(candidate)

        return mutated_cand


    def mutationSelectionGuidedRandom(self, candidate):
        pass



    def mutationSelectionNN(self, candidate, choice):
        if choice == 0:
            mutated_cand = self.swap(candidate)
        elif choice == 1:
            mutated_cand = self.flip(candidate)
        elif choice == 2:
            mutated_cand = self.regenerate(candidate)
        elif choice == 3:
            mutated_cand = self.smartRegenerate(candidate)
        elif choice == 4:
            mutated_cand = self.normalise(candidate)
        elif choice == 5:
            mutated_cand = self.FilteredMutation(candidate)
        candidate.prev_mutation = choice
        return mutated_cand



    #---------------------------------------GENERATION VECTORISER---------------------------------------------------
    def make_fitness_histogram(self):
        fitnesses = [grid.fitness for grid in self.population]
        histogram = Counter(fitnesses)
        return histogram


#---------------------------------------MAIN--------------------------------------------------------------

def main():
    parents = []
    EA = EvolutionaryAlgorithm()
    #Initialise the population
    EA.GenPop(sample_grid)
    gen = 0
    while True:
        #print(f"Generation = {gen}, best solution has fitness = {EA.population[0].fitness}")
        #select the best candidates and generate new generation
        parents = EA.selectionTournament()

        #generate offspring
        for parent in range(0, len(parents), 2):
            offspring1, offspring2 = EA.crossoverCyclic(parents[parent].grid, parents[parent+1].grid, parents[parent].constrained_grid)
            #randomly mutate some offspring
            mutation_bound_1 = random()
            mutation_bound_2 = random()
            if mutation_bound_1 < 0.2:
                EA.mutationSelectionRandom(offspring1)
            if mutation_bound_2 < 0.2:
                EA.mutationSelectionRandom(offspring2)
            #add offspring to population
            EA.population.append(offspring1)
            EA.population.append(offspring2)
        #examine population and prune
        #replacement - use elitism
        EA.Replacement()
        #continue until termination is met.
        if EA.population[0].fitness == 0:
            print("solution found")
            return EA

        gen += 1







#---------------------------------------DATA ACQUISITION FOR TRAINING--------------------------------------------------------------

#function to produce a set of data entries for the neural network
def getRunData(grid):
    #same as main with data extraction
    EA_sample = EvolutionaryAlgorithm()
    EA_sample.initialise(grid)
    new_data, constrained_grid = EA_sample.runWithDataAcquistion()
    #extract information
    return new_data, constrained_grid



def getManyRunData(grid, no_of_runs = 10):
    all_data = []
    constrained_grid = []
    for i in range(no_of_runs):
        new_d, constrained_grid = getRunData(grid)
        all_data+=new_d
    return all_data, constrained_grid


def getRunDataTraining(grid):
    #same as main with data extraction
    EA_sample = EvolutionaryAlgorithm()
    POP_SIZE = 1
    EA_sample.initialise(grid)
    new_data, constrained_grid = EA_sample.runWithDataAcquistionTraining()
    #extract information
    return new_data, constrained_grid


def getManyRunDataTraining(grid, no_of_runs = 10):
    all_data = []
    constrained_grid = []
    for i in range(no_of_runs):
        new_d, constrained_grid = getRunDataTraining(grid)
        all_data+=new_d
    return all_data, constrained_grid



def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError


def storeData(constrained_grid, all_data, problem_name):
    #locate folder to store in, within "DATA/PROBLEM/"
    #store in DS
    path = PATH_TO_DATA + f"/{problem_name}/"

    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    #first, check if file for constrained grid exists, if not, make it, if so, pass
    constrained_grid_file = f"constrained_grid_{problem_name}"    
    if constrained_grid_file not in os.listdir(path):
        #write constrained_grid to file as json
        with open(os.path.join(path, constrained_grid_file), 'w') as f:
            json.dump(constrained_grid, f)

    # Determine the starting index based on the number of existing files
    starting_index = len(os.listdir(path))

    # Store data in chunks of 1000
    for chunk_index, i in tqdm(enumerate(range(0, len(all_data), 1000), start=starting_index)):
        chunk = all_data[i:i + 1000]
        file_name = f"{problem_name}_training_set_{chunk_index}.json"
        with open(os.path.join(path, file_name), 'w') as f:
            try:
                json.dump(chunk, f, default=convert_numpy)
            except TypeError as e:
                print(f"Error serializing chunk {chunk_index}: {e}")


#reads a file
def readJsonFile(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

#alter to read a subset of files in a folder
def readFiles(problem_name):
    #locate folder to store in, within "DATA/PROBLEM/"
    #store in DS
    path = PATH_TO_DATA + f"/{problem_name}/"
    #locate all files in the folder
    files = os.listdir(path)
    #read all files
    data = []
    for file in files:
        data.append(readJsonFile(file))
    return data


#-----------------------------------------------LOADING GRIDS------------------------------------------------

def loadGrid(problem_name):
    path = PATH_TO_GRIDS + f"/{problem_name}.txt"
    with open(path  , 'r') as file:
        #convert to data structure
        grid_str = file.read()
        grid = ast.literal_eval(grid_str)
        return grid






'''
prob_types = ["easy", "hard", "medium", "veryhard"]
prob_nums = [1,2,3]
for prob in tqdm(prob_types, desc='Problem Types'):
    for num in tqdm(prob_nums, desc='Problem Numbers', leave=False):
        grid = loadGrid(f"{prob}{num}")
        all_data, constrained_grid = getManyRunDataTraining(grid, 10)
        storeData(constrained_grid, all_data, f"{prob}{num}")


'''

#test new mutation filtered mutation

'''
grid = loadGrid("easy1")
EA = EvolutionaryAlgorithm(grid, None, None, None)
EA.population = [EA.generateCandidate(grid)]
print(EA.population[0].grid)
EA.FilteredMutation(EA.population[0])
print(EA.population[0].grid)'''