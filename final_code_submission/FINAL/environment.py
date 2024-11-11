import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate, LSTM
from keras.optimizers import Adam
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from GA_main import EvolutionaryAlgorithm, Grid
import ast
from keras.utils import plot_model
from model import SudokuModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

script_directory = os.path.dirname(os.path.realpath(__file__))
PATH_TO_GRIDS = os.path.join(script_directory,'GRIDS')




#reinforcement learning
num_episodes = 100
optimizer = Adam(learning_rate=0.001)
max_steps_per_episode = 100  # Adjust as needed


class SudokuEnvironment:
    def __init__(self, ea, initial_grid, model, type):
        self.ea = ea
        self.current_state = None
        self.model = model
        self.type = type


    #generational step
    def step(self, termination_value, update_interval = 1):
        if self.type == 0:
            self.ea.run_generation_2(termination_value)
        else:
            self.ea.run_generation_guided(termination_value, update_interval)


    def run(self, termination_value, info_extraction_function = None, info_extraction_function2 = None, info_extraction_function3 = None, update_interval = 1):
        data = []
        data2 = []
        data3 = []
        while True:
            print(f"g {self.ea.generation}")
            self.step(termination_value, update_interval)
            #extract generation data
            population = self.ea.population
            if info_extraction_function == extract_sum_reward:
                data.append(info_extraction_function(self.ea.class_rewards_copy))
            elif info_extraction_function == extract_mutations_probs:
                data.append(info_extraction_function(population[0], self.model, self.ea.generation, self.ea.iterations_since_best_fitness))
            else:
                if info_extraction_function:
                    data.append(info_extraction_function(population))
                if info_extraction_function2:
                    data2.append(info_extraction_function2(population))
                if info_extraction_function3:
                    data3.append(info_extraction_function3(population))
            #check termination criteria
            if self.ea.population[0].fitness == 0 or self.ea.generation == termination_value:
                return data, data2, data3
            #input("end")
            
        


def loadGrid(problem_name):
    path = PATH_TO_GRIDS + f"/{problem_name}.txt"
    with open(path  , 'r') as file:
        #convert to data structure
        grid_str = file.read()
        grid = ast.literal_eval(grid_str)
        return grid




#--------------------------------------------DATA EXTRACTION FUNCTIONS--------------------------------------------
def extract_best_fitness(population):
    return population[0].fitness


def extract_fitness_diversity(population):
    fitnesses = [individual.fitness for individual in population]
    return np.std(fitnesses)

def extract_avg_fitness(population):
    fitnesses = [individual.fitness for individual in population]
    return np.mean(fitnesses)


def extract_sum_reward(rewards):
    # Assuming 'rewards' is a dictionary with lists of numbers as values
    total = 0
    for reward_list in rewards.values():
        total += sum(reward_list)  # Sum each list individually
    return total

def extract_mutations_probs(cand, model, generation, iterations_since_best_fitness):
    grid_tensor = np.array(cand.grid).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
    scalar_inputs = np.array([cand.fitness, cand.prev_mutation, generation, iterations_since_best_fitness])
    # Predict mutation action and apply
    action_probs = model.model([grid_tensor, scalar_inputs.reshape(1, -1)])
    return action_probs


#-------------------------------------------PLOTTING--------------------------------------------------------



def plot_fitness(data, title, y_label):
    """
    Plots the best fitness per generation using the index of each fitness value as the generation number.

    Parameters:
    fitnesses (list of float): The best fitness values for each generation.
    """
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(data, linestyle='-', color='b')  # Plot with blue line and circle markers
    plt.title(title)  # Title of the plot
    plt.xlabel("Generation")  # X-axis label
    plt.ylabel(y_label)  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
    plt.show()  # Display the plot




def plot_fitness_and_diversity(best_fitnesses, average_fitnesses, diversities):
    """
    Plots the best and average fitness values, a line of best fit for the best fitness values, and diversity per generation using the index of each list item as the generation number.

    Parameters:
    best_fitnesses (list of float): The best fitness values for each generation.
    average_fitnesses (list of float): The average fitness values for each generation.
    diversities (list of float): The diversity measurements for each generation.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))  # Create a figure and an axis for fitness

    # Plotting average fitness on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness', color=color)
    ax1.plot(average_fitnesses, color=color, linestyle='-', label='Average Fitness')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plotting best fitness
    ax1.plot(best_fitnesses, color='tab:purple', linestyle='--', label='Best Fitness')

    # Calculating and plotting the line of best fit for the best fitness values
    generations = np.arange(len(best_fitnesses))  # Assuming generations are sequential and start from 0
    slope, intercept = np.polyfit(generations, best_fitnesses, 1)
    #best_fit_line = slope * generations + intercept
    #ax1.plot(generations, best_fit_line, color='tab:green', linestyle=':', label='Line of Best Fit')
    ax1.legend(loc='upper right')  # Adding legend to the plot

    ax1.grid(True)

    # Creating a secondary y-axis for diversity
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Diversity', color=color)  # We already handled the x-label with ax1
    ax2.plot(diversities, color=color, linestyle='--', label='Diversity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper left')  # Adding legend to the plot

    # Title of the plot
    plt.title("Fitness and Diversity per Generation")

    # Show the plot
    plt.show()





def selection_test(selectionMethods, problems, num_of_runs):
    for selectionMethod in selectionMethods:
        selection_method_data = []
        print("e")
        for problem in problems:
            problem_data = []
            for i in range(num_of_runs):
                grid = loadGrid(problem)
                model = SudokuModel(num_mutation_types=6)
                EA = EvolutionaryAlgorithm(grid=grid, guided_bool=True, model=model)

                if selectionMethod == EvolutionaryAlgorithm.selectionTournament:
                    EA.selection_method = EA.selectionTournament
                elif selectionMethod == EvolutionaryAlgorithm.selectionRouletteWheel:
                    EA.selection_method = EA.selectionRouletteWheel
                else:
                    EA.selection_method = EA.selectionRankBased

                EA.mutation_selecter = EA.mutationSelectionRandom2
                EA.crossover_method = EA.crossoverCyclic
                environment = SudokuEnvironment(EA, grid, model, 1)
                data, data2, data3 = environment.run(300, extract_avg_fitness, extract_fitness_diversity, extract_best_fitness, 5)

                # reward if solution found - show fitness spread, average, best, and worst. show diversity 
                #append best fitness and average diversity to problem data
                problem_data.append([min(data3), sum(data2) / len(data2), EA.generation])
            #extract useful data from problem_data
            fitnesses = [x[0] for x in problem_data]
            avg_diversity = sum([x[1] for x in problem_data]) / len(problem_data)
            avg_generations = sum([x[2] for x in problem_data]) / len(problem_data)
            
            selection_method_data.append([fitnesses, avg_diversity, avg_generations])
    return selection_method_data    




def crossover_test(crossover_methods, problems, num_of_runs):
    all_selection_meth_data = []
    for selectionMethod in crossover_methods:
        selection_method_data = []
        print("e")
        for problem in problems:
            problem_data = []
            for i in range(num_of_runs):
                grid = loadGrid(problem)
                model = SudokuModel(num_mutation_types=6)
                EA = EvolutionaryAlgorithm(grid=grid, guided_bool=True, model=model)

                if selectionMethod == EvolutionaryAlgorithm.crossoverAlternateRows:
                    EA.crossover_method = EA.crossoverAlternateRows
                elif selectionMethod == EvolutionaryAlgorithm.crossoverCyclic:
                    EA.crossover_method = EA.crossoverCyclic
                else:
                    EA.crossover_method = EA.crossoverPMX
                EA.selection_method = None #YET TO BE OPTIMISED
                EA.mutation_selecter = EA.mutationSelectionRandom2
                environment = SudokuEnvironment(EA, grid, model, 1)
                data, data2, data3 = environment.run(300, extract_avg_fitness, extract_fitness_diversity, extract_best_fitness, 5)

                # reward if solution found - show fitness spread, average, best, and worst. show diversity 
                #append best fitness and average diversity to problem data
                problem_data.append([min(data3), sum(data2) / len(data2), EA.generation])
            #extract useful data from problem_data
            fitnesses = [x[0] for x in problem_data]
            avg_diversity = sum([x[1] for x in problem_data]) / len(problem_data)
            avg_generations = sum([x[2] for x in problem_data]) / len(problem_data)
            
            selection_method_data.append([fitnesses, avg_diversity, avg_generations])
        all_selection_meth_data.append([selection_method_data])
    return selection_method_data    




def learning_rate_test(learning_rates, problems, num_of_runs):
    l_r_data = []
    for l_r in learning_rates:
        all_prob_data = []
        for problem in problems:
            problem_data = []
            for i in range(num_of_runs):
                grid = loadGrid(problem)
                model = SudokuModel(num_mutation_types=6)
                EA = EvolutionaryAlgorithm(grid=grid, guided_bool=True, model=model)
                EA.mutation_selecter = EA.mutationSelectionRandom2
                EA.crossover_method = EA.crossoverCyclic
                environment = SudokuEnvironment(EA, grid, model, 1)
                data, data2, data3 = environment.run(150, extract_avg_fitness, extract_fitness_diversity, extract_best_fitness, l_r)
                best_fitness = min(data3)
                avg_diversity = sum(data2) / len(data2)
                problem_data.append([best_fitness, avg_diversity])
            all_prob_data.append([problem_data])
        #add to L_R data
        l_r_data.append([all_prob_data])
    return l_r_data           


def reward_over_time_test(problems, num_of_runs):
    #rewards over time
    rewards = []
    for problem in problems:
        for i in range(num_of_runs):
            grid = loadGrid(problem)
            model = SudokuModel(num_mutation_types=6)
            EA = EvolutionaryAlgorithm(grid=grid, guided_bool=True, model=model)
            EA.mutation_selecter = EA.mutationSelectionRandom2
            EA.selection_method = EA.selectionRouletteWheel
            EA.crossover_method = EA.crossoverCyclic
            environment = SudokuEnvironment(EA, grid, model, 1)
            data, data2, data3 = environment.run(200, extract_sum_reward)
            print(data)
            rewards.append(data)
    return rewards  






def plot_fitness_violins(data, methods, difficulties):
    all_data = []
    for method_index, method in enumerate(methods):
        for difficulty_index, difficulty in enumerate(difficulties):
            index = method_index * len(difficulties) + difficulty_index
            fitnesses = data[index][0]  # Extract list of fitnesses
            diversity = data[index][1]  # Extract diversity value

            for fitness in fitnesses:
                all_data.append({
                    'Fitness': fitness,
                    'Method': method,
                    'Difficulty': difficulty,
                    'Diversity': diversity
                })

    df = pd.DataFrame(all_data)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Difficulty', y='Fitness', hue='Method', data=df, palette='muted', split=True, inner='quartile')
    ax.set_title('Performance of Crossover Methods on Variable Puzzle Difficulties')
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Difficulty Level')
    plt.legend(title='Selection Method', loc='lower right')

    # Annotate diversity values with vertical adjustment
    unique_combinations = df.groupby(['Method', 'Difficulty'])
    for (method, difficulty), group in unique_combinations:
        diversity = group['Diversity'].values[0]  # Assuming the same for all entries in group
        # Adjust the y-position to lower the annotations
        y_position = ax.get_ylim()[0] + 0.95 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        x_position = difficulties.index(difficulty) + methods.index(method) * 0.2 - 0.15
        plt.text(x_position, y_position, f'Div: {diversity:.2f}', horizontalalignment='center', verticalalignment='top', rotation=45)

    plt.show()


def plot_fitness_change_interval(reorganized_data, learning_rates, problems):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    ax2 = ax1.twinx()  # Create a secondary y-axis
    
    # Colors for different problems
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    
    for problem_idx, problem_data in enumerate(reorganized_data):
        problem_name = problems[problem_idx]  # Use the provided problem names
        intervals_data = problem_data[1]
        avg_fitness_per_lr = [interval[1][0] for interval in intervals_data]
        avg_diversity_per_lr = [interval[1][1] for interval in intervals_data]
        std_dev_per_lr = [interval[1][2] for interval in intervals_data]

        # Plot the average fitness for this problem across all learning rates
        ax1.plot(learning_rates, avg_fitness_per_lr, label=f'{problem_name} Fitness', color=colors[problem_idx % len(colors)])
        
        # Plot the standard deviation bounds for fitness
        lower_bound_fitness = [avg - std for avg, std in zip(avg_fitness_per_lr, std_dev_per_lr)]
        upper_bound_fitness = [avg + std for avg, std in zip(avg_fitness_per_lr, std_dev_per_lr)]
        ax1.fill_between(learning_rates, lower_bound_fitness, upper_bound_fitness, color=colors[problem_idx % len(colors)], alpha=0.2)

        # Plot the average diversity for this problem across all learning rates on secondary y-axis
        ax2.plot(learning_rates, avg_diversity_per_lr, label=f'{problem_name} Diversity', linestyle='dashed', color=colors[problem_idx % len(colors)])
    
    # Setting labels and title
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Average Best Fitness')
    ax2.set_ylabel('Average Diversity')
    ax1.set_title('Average Best Fitness and Diversity per Reinforcement Update Interval')
    
    # Combine legends from both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.show()





def process_and_reorganize_data(l_r_data):
    data = []

    for l_r in l_r_data:
        l_r_actual = l_r[0]
        problem_data_list = []
        
        for problem_data in l_r_actual:
            problem_data_actual = problem_data[0]

            average_fitness = sum([x[0] for x in problem_data_actual]) / len(problem_data_actual)
            average_diversity = sum([x[1] for x in problem_data_actual]) / len(problem_data_actual)
            std_dev_fitness = np.std([x[0] for x in problem_data_actual])
            problem_data_list.append([average_fitness, average_diversity, std_dev_fitness])
        
        data.append(problem_data_list)

    # Rearrange the data into the desired format
    reorganized_data = []

    for problem_idx in range(len(data[0])):  # Iterate over problems
        puzzle_data = []


        for lr_idx in range(len(data)):  # Iterate over learning rates
            interval_data = [lr_idx + 1, data[lr_idx][problem_idx]]

            puzzle_data.append(interval_data)


        reorganized_data.append([f'Problem {problem_idx + 1}', puzzle_data])

    return reorganized_data



#mutation prob to generations
#get distribution for current best solution - because it shows how well the algorithm is doing on a consistent cand (after plateau)
#end of each gen, get best solution, feed to the NN, get distribution and save

def mutation_prob_over_gens(problem, interval):

    grid = loadGrid(problem)
    model = SudokuModel(num_mutation_types=6)
    EA = EvolutionaryAlgorithm(grid=grid, guided_bool=True, model=model)
    EA.selection_method = EA.selectionRouletteWheel
    EA.mutation_selecter = EA.mutationSelectionRandom2
    EA.crossover_method = EA.crossoverCyclic
    environment = SudokuEnvironment(EA, grid, model, 1)
    data, _, _ = environment.run(100, extract_mutations_probs, interval)
    best_fitness = EA.population[1].fitness

    return data, best_fitness


def plot_class_probabilities_over_generations(mute_data):
    # Ensure data is in the correct format (list of lists)
    mute_data = np.array(mute_data).squeeze()
    
    # Extract generations and class probabilities
    generations = list(range(mute_data.shape[0]))
    class_probs = mute_data.T  # Transpose to get probabilities for each class over generations

    # Verify dimensions
    print(f"Generations: {len(generations)}")
    print(f"Class Probabilities: {class_probs.shape}")

    # Define colors for each class
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.stackplot(generations, class_probs, labels=["Swap", "Flip", "Regenerate", "SmartRegenerate", "Normalise", "Filtered Mutation"], colors=colors[:class_probs.shape[0]], alpha=0.6)
    
    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Class Probabilities')
    plt.title('Class Probabilities Over Generations')
    plt.legend(loc='upper left')
    
    # Display the plot
    plt.show()



def plot_final_performance(guided_data, normal_data):
    # Convert lists of lists to numpy arrays for easier manipulation
    guided_data = [np.array(run_data) for run_data in guided_data]
    normal_data = [np.array(run_data) for run_data in normal_data]
    
    # Stack the arrays for each metric across runs
    guided_avg_fitness = np.stack([run[0] for run in guided_data])
    guided_diversity = np.stack([run[1] for run in guided_data])
    guided_best_fitness = np.stack([run[2] for run in guided_data])
    
    normal_avg_fitness = np.stack([run[0] for run in normal_data])
    normal_diversity = np.stack([run[1] for run in normal_data])
    normal_best_fitness = np.stack([run[2] for run in normal_data])
    
    # Calculate mean and standard deviation across runs for each metric
    guided_avg_fitness_mean = np.mean(guided_avg_fitness, axis=0)
    guided_avg_fitness_std = np.std(guided_avg_fitness, axis=0)
    guided_diversity_mean = np.mean(guided_diversity, axis=0)
    guided_best_fitness_mean = np.mean(guided_best_fitness, axis=0)
    guided_best_fitness_std = np.std(guided_best_fitness, axis=0)
    
    normal_avg_fitness_mean = np.mean(normal_avg_fitness, axis=0)
    normal_avg_fitness_std = np.std(normal_avg_fitness, axis=0)
    normal_diversity_mean = np.mean(normal_diversity, axis=0)
    normal_best_fitness_mean = np.mean(normal_best_fitness, axis=0)
    normal_best_fitness_std = np.std(normal_best_fitness, axis=0)
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()
    
    # Plotting the average fitness with error bounds
    ax1.plot(guided_avg_fitness_mean, label='Guided Avg Fitness', linestyle='-')
    ax1.fill_between(range(len(guided_avg_fitness_mean)), 
                     guided_avg_fitness_mean - guided_avg_fitness_std, 
                     guided_avg_fitness_mean + guided_avg_fitness_std, 
                     alpha=0.2)
    
    ax1.plot(normal_avg_fitness_mean, label='Normal Avg Fitness', linestyle='-')
    ax1.fill_between(range(len(normal_avg_fitness_mean)), 
                     normal_avg_fitness_mean - normal_avg_fitness_std, 
                     normal_avg_fitness_mean + normal_avg_fitness_std, 
                     alpha=0.2)
    
    # Plotting the best fitness with error bounds
    ax1.plot(guided_best_fitness_mean, label='Guided Best Fitness', linestyle='--')
    ax1.fill_between(range(len(guided_best_fitness_mean)), 
                     guided_best_fitness_mean - guided_best_fitness_std, 
                     guided_best_fitness_mean + guided_best_fitness_std, 
                     alpha=0.2)
    
    ax1.plot(normal_best_fitness_mean, label='Normal Best Fitness', linestyle='--')
    ax1.fill_between(range(len(normal_best_fitness_mean)), 
                     normal_best_fitness_mean - normal_best_fitness_std, 
                     normal_best_fitness_mean + normal_best_fitness_std, 
                     alpha=0.2)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness')
    ax1.legend(loc='upper left')
    
    # Plotting the diversity on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(guided_diversity_mean, label='Guided Diversity', linestyle=':', color='red')
    ax2.plot(normal_diversity_mean, label='Normal Diversity', linestyle=':', color='blue')
    ax2.set_ylabel('Diversity')
    ax2.legend(loc='upper right')
    
    # Show the plot
    plt.title('Performance Comparison between Guided and Normal EA')
    plt.show()




def plot_final_performance(guided_data, normal_data, title):
    # Assuming each run is structured as [avg_fitness_list, diversity_list, best_fitness_list]
    guided_avg_fitness = []
    guided_diversity = []
    guided_best_fitness = []
    for run in guided_data:
        guided_avg_fitness.append(run[0])
        guided_diversity.append(run[1])
        guided_best_fitness.append(run[2])

    normal_avg_fitness = []
    normal_diversity = []
    normal_best_fitness = []
    for run in normal_data:
        normal_avg_fitness.append(run[0])
        normal_diversity.append(run[1])
        normal_best_fitness.append(run[2])

    def calculate_bounds(data):
        max_length = max(len(run) for run in data)
        padded_data = [run + [None] * (max_length - len(run)) for run in data]
        avg = [sum(x for x in vals if x is not None) / len([x for x in vals if x is not None]) for vals in zip(*padded_data)]
        best = [max(x for x in vals if x is not None) for vals in zip(*padded_data)]
        worst = [min(x for x in vals if x is not None) for vals in zip(*padded_data)]
        return avg, best, worst

    # Calculate the average, best, and worst for guided and normal data
    guided_avg_fitness_mean, guided_avg_fitness_best, guided_avg_fitness_worst = calculate_bounds(guided_avg_fitness)
    guided_best_fitness_mean, guided_best_fitness_best, guided_best_fitness_worst = calculate_bounds(guided_best_fitness)
    guided_diversity_mean, _, _ = calculate_bounds(guided_diversity)
    
    normal_avg_fitness_mean, normal_avg_fitness_best, normal_avg_fitness_worst = calculate_bounds(normal_avg_fitness)
    normal_best_fitness_mean, normal_best_fitness_best, normal_best_fitness_worst = calculate_bounds(normal_best_fitness)
    normal_diversity_mean, _, _ = calculate_bounds(normal_diversity)
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()
    
    # Plotting the average fitness with bounds
    ax1.plot(guided_avg_fitness_mean, label='Guided Avg Fitness', linestyle='-')
    ax1.fill_between(range(len(guided_avg_fitness_mean)), 
                     guided_avg_fitness_worst, 
                     guided_avg_fitness_best, 
                     alpha=0.2)
    
    ax1.plot(normal_avg_fitness_mean, label='Normal Avg Fitness', linestyle='-')
    ax1.fill_between(range(len(normal_avg_fitness_mean)), 
                     normal_avg_fitness_worst, 
                     normal_avg_fitness_best, 
                     alpha=0.2)
    
    # Plotting the best fitness with bounds
    ax1.plot(guided_best_fitness_mean, label='Guided Best Fitness', linestyle='--')
    ax1.fill_between(range(len(guided_best_fitness_mean)), 
                     guided_best_fitness_worst, 
                     guided_best_fitness_best, 
                     alpha=0.2)
    
    ax1.plot(normal_best_fitness_mean, label='Normal Best Fitness', linestyle='--')
    ax1.fill_between(range(len(normal_best_fitness_mean)), 
                     normal_best_fitness_worst, 
                     normal_best_fitness_best, 
                     alpha=0.2)
    
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness')
    ax1.legend(loc='upper left')
    
    # Plotting the diversity on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(guided_diversity_mean, label='Guided Diversity', linestyle=':', color='red')
    ax2.plot(normal_diversity_mean, label='Normal Diversity', linestyle=':', color='blue')
    
    ax2.set_ylabel('Diversity')
    ax2.legend(loc='upper right')
    
    # Show the plot
    plt.title(title)
    plt.show()

# Example usage:
# guided_data, normal_data = compare_performance(problem, iterations)
# plot_final_performance(guided_data, normal_data)


#-------------------------------------------SAVE AND LOAD--------------------------------------------------------

def save_results_to_json(data, filename="selection_results_final.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results have been saved to {filename}")





def extract_values(tensors):
    extracted_values = []
    for tensor in tensors[0]:
        values = tensor.numpy().tolist()[0]
        extracted_values.append(values)
    return extracted_values



#load guided and normal EA data

data = json.load(open("guided_vs_normal_veryhard1(1).json", "r"))
guided_data = data[0]
normal_data = data[1]




#count instances in both where no. of gens is less than 150
guided_count = 0
normal_count = 0
for instance in range(len(normal_data)):
    if len(guided_data[instance][0]) < 150:
        guided_count+=1
    if len(normal_data[instance][0]) < 150:
        normal_count+=1


plot_final_performance(guided_data, normal_data, f'Performance Comparison between GEA and EA on very hard puzzles, GEA finished {guided_count} instances early, EA finished {normal_count} instances early')


'''
#load from json
data = json.load(open("guided_vs_normal.json", "r"))

guided_data = data[0]
normal_data = data[1]

plot_final_performance(guided_data, normal_data)
'''


'''#run guided and normal EA on a problem, compare performance
guided_data, normal_data = compare_performance("easy1", 1)


save_results_to_json([guided_data, normal_data], "guided_vs_normal.json")

'''

'''interval = 1
mute_data = mutation_prob_over_gens("hard1", interval)

converted_mute_data = extract_values(mute_data)

# Save results to JSON
#save_results_to_json(converted_mute_data, f"mutation_prob_over_gens_{interval}.json")

plot_class_probabilities_over_generations(converted_mute_data)
'''



'''

#load from josn
data = json.load(open("crossover_results_final_REAL.json", "r"))
data_alt = data[0]
data_cyc = data[1]
datapmx = json.load(open("crossover_results_final_PMX.json", "r"))


data = data_alt + data_cyc + datapmx

print(len(data))
print(data)
input()

plot_fitness_violins(data, ["Alternate Rows", "Cyclic", "PMX"], ["easy1", "easy2", "medium1", "medium2", "hard1", "hard2"])
'''
'''
selection_data = selection_test([EvolutionaryAlgorithm.selectionRouletteWheel], ["easy1", "easy2", "medium1", "medium2", "hard1", "hard2"], 5)
# old one contains , EvolutionaryAlgorithm.selectionRankBased, , EvolutionaryAlgorithm.selectionTournament
print(selection_data)
save_results_to_json(selection_data, "selection_results_final_3.json")
'''




'''

intervals = json.load(open("L_R_results_final.json", "r"))

reorganised_data = process_and_reorganize_data(intervals)

plot_fitness_change_interval(reorganised_data, [1, 2, 3, 4, 5, 6, 7], ["easy", "medium", "hard"])
'''


#reward_over_time_data = reward_over_time_test(["easy1"],1)
#print(reward_over_time_data)
#input()

'''reward_over_time_data = reward_over_time_test(["easy1", "easy2", "medium1", "medium2", "hard1", "hard2"], 3)
save_results_to_json(reward_over_time_data, "reward_over_time_data_final.json")
'''


#for learning curve, do intervals 1 to 7, for 3 problems, averaged over 5 runs
