import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate, LSTM
from keras.optimizers import Adam
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from FINAL.GA_main import EvolutionaryAlgorithm
import ast

from keras.utils import plot_model


script_directory = os.path.dirname(os.path.realpath(__file__))
PATH_TO_GRIDS = os.path.join(script_directory, '..', 'Implementation', 'GRIDS')




# Number of mutation types for classification
num_mutation_types = 6  # 6 mutation types + 1 for no mutation

def makeModel():
    # Assuming the grid is a 9x9 Sudoku grid, reshape it for CNN input
    grid_shape = (9, 9, 1)  # 9x9 grid with 1 channel


    # CNN branch for grid
    grid_input = Input(shape=grid_shape, name='grid_input')
    cnn = Conv2D(32, (3, 3), activation='relu')(grid_input)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = Flatten()(cnn)

    # Dense branch for scalar inputs
    scalar_input = Input(shape=(4,), name='scalar_input')  # Grid fitness, current generation, generations since best fitness
    dense = Dense(16, activation='relu')(scalar_input)

    # Concatenate the outputs from both branches
    combined = concatenate([cnn, dense])


    # Modify the output layer for self-supervised learning
    output = Dense(num_mutation_types, activation='softmax')(combined)  # Predicting the prev_mutation as a regression problem

    # Create the model for self-supervised learning
    self_supervised_model = Model(inputs=[grid_input, scalar_input], outputs=output)

    # Compile the model for self-supervised learning
    self_supervised_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # Use MSE if predicting prev_mutation as a continuous value

    # Summary of the model
    self_supervised_model.summary()

    return self_supervised_model   













#reinforcement learning
num_episodes = 100
optimizer = Adam(learning_rate=0.001)
max_steps_per_episode = 100  # Adjust as needed


class SudokuEnvironment:
    def __init__(self, initial_grid, model):
        self.ea = EvolutionaryAlgorithm()
        self.ea.initialise(initial_grid)
        self.current_state = None
        self.model = model
        self.reset(initial_grid)

    def reset(self, grid):
        self.ea = EvolutionaryAlgorithm()
        self.ea.initialise(grid)  # Use your preferred starting grid
        self.current_state = self.ea.population[0].grid
        return np.array(self.current_state, dtype=np.float32).reshape(1, 9, 9, 1)  # Reshape to the expected input shape


    def step(self, action):
        initial_fitness = self.ea.population[0].fitness
        
        # Use the action to determine which mutation to apply
        if action == 0:
            pass  # Do nothing
        elif action == 1:
            self.ea.swap(self.ea.population[0])
        elif action == 2:
            self.ea.flip(self.ea.population[0])
        elif action == 3:
            self.ea.regenerate(self.ea.population[0])
        elif action == 4:
            self.ea.smartRegenerate(self.ea.population[0])
        elif action == 5:
            self.ea.normalise(self.ea.population[0])
        elif action == 6:
            self.ea.smartSquare(self.ea.population[0])
        else:
            self.ea.mutationSelectionRandom(self.ea.population[0])

        # assign prev_mutation
        self.ea.population[0].prev_mutation = action

        self.ea.Replacement()

        # Prepare grid tensor and scalar inputs
        grid_tensor = np.array(self.current_state).reshape(1, 9, 9, 1)  # Ensure it's float32 if needed
        scalar_inputs = np.array([
            [self.ea.population[0].fitness, self.ea.population[0].prev_mutation, self.ea.generation, self.ea.iterations_since_best_fitness]
        ], dtype=np.float32)

        # Correctly pass inputs to the model
        action_probs = self.model([grid_tensor, scalar_inputs], training=True)

        # Calculate reward
        reward = initial_fitness - self.ea.population[0].fitness

        # Update current state
        self.current_state = np.array(self.ea.population[0].grid, dtype=np.float32).reshape(1, 9, 9, 1)

        return self.current_state, reward, action_probs


def loadGrid(problem_name):
    path = PATH_TO_GRIDS + f"/{problem_name}.txt"
    with open(path  , 'r') as file:
        #convert to data structure
        grid_str = file.read()
        grid = ast.literal_eval(grid_str)
        return grid



# Assume `model` is your pre-defined model
#get instance of a grid
prob = "easy"
num = "1"
grid = loadGrid(f"{prob}{num}")
model = makeModel()

environment = SudokuEnvironment(grid, model)
print(environment.ea.population[0].grid)

#make the model
model = SudokuModel(num_mutation_types)



'''
input("ready to visualise?") 
model = makeModel()  # Assuming your model is built in a function
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

input("training time")

for episode in range(num_episodes):
    with tf.GradientTape() as tape:
        state = environment.reset(grid)  # Reset Sudoku puzzle
        scalar_inputs = np.array([environment.ea.population[0].fitness, environment.ea.population[0].prev_mutation, environment.ea.generation, environment.ea.iterations_since_best_fitness], dtype=np.float32).reshape(1, -1)
        episode_reward = 0

        for timestep in range(max_steps_per_episode):
            # Model predicts the action probabilities
            print("State shape:", state.shape)  # Should be (1, 9, 9, 1)
            print("Scalar inputs shape:", scalar_inputs.shape)  # Should be (1, 4)

            print(f"state: {state}, scalar_inputs: {scalar_inputs}")

            action_probs = model([state, scalar_inputs], training=True)
            action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)[0,0]
            print(f"action_probs: {action_probs}, action: {action}")
            input()
            
            # Apply the chosen action to the environment
            next_state, reward, action_probs = environment.step(action.numpy())
            
            # Compute loss (negative log probability of the action, scaled by the reward)
            action_prob = action_probs[0, action]
            loss = -tf.math.log(action_prob) * reward
            print(loss)

            episode_reward += reward
            state = next_state
            scalar_inputs = np.array([[environment.ea.population[0].fitness, environment.ea.population[0].prev_mutation, environment.ea.generation, environment.ea.iterations_since_best_fitness]]) 

        # Compute gradients and update model weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f'Episode {episode}, Total Reward: {episode_reward}')
'''