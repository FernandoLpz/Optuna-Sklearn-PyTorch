# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import optuna

class NeuralNet(nn.ModuleList):
    def __init__(self, trial):
        super(NeuralNet, self).__init__()

        # Where the layers and dropouts will be stacked
        self.layers = []
        self.dropouts = []

        # Given a trial, takes the suggested number of 
        # layers as well as the dropout value  
        n_layers = trial.suggest_int('n_layers', 1,4)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

        # Since the input tensor has a shape (n, 30)
        input_dim = 30

        # Given the suggested number of layers, proceeds by
        # stacking layer by layer. Likewise the number of internal
        # units is a value taken from the current trial suggestion
        for layer in range(n_layers):
            output_dim = trial.suggest_int(f"output_dim", 4, 30)
            
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(dropout)

            input_dim = output_dim
        
        # The last layer is added to the stack. Sice we are working
        # with binary classification, 1 output unit is enough
        self.layers.append(nn.Linear(input_dim, 1))

        # PyTorch needs each layer, dropout, etc to be defined as an instance 
        # variable of the class. E.g. self.layer_1 = nn.Linear(in, out)
        # Thus we need to create such instance variable from the list of stacked 
        # layers and dropouts with the "setattr" method.
        for idx, layer in enumerate(self.layers):
            setattr(self, f"fc_{idx}", layer)

        for idx, dropout in enumerate(self.dropouts):
            setattr(self, f"dr_{idx}", dropout)

    def forward(self, x):
        # Traverse each layer & dropout
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = F.relu(x)
            x = dropout(x)

        # Last layer has sigmoid as 
        # activation funciton
        x = torch.sigmoid(x)

        return x.squeeze()

if __name__ == '__main__':

    # Load data and split into train and test sets
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=23)

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")