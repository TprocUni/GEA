import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from keras.optimizers import Adam

class SudokuModel:
    def __init__(self, num_mutation_types=6):
        # Number of mutation types for classification: 6 mutation types + 1 for no mutation
        self.num_mutation_types = num_mutation_types
        self.model = self._make_model()
        self.optimizer = Adam(learning_rate=0.001)

    def _make_model(self):
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

        # Output layer for self-supervised learning
        output = Dense(self.num_mutation_types, activation='softmax')(combined)

        # Creating the model for self-supervised learning
        model = Model(inputs=[grid_input, scalar_input], outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Returning the model
        return model

    def summary(self):
        # Print a summary of the model
        return self.model.summary()

    def get_model(self):
        # Accessor for the model
        return self.model
