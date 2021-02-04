import os
import optuna

# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST


class NeuralNet(nn.ModuleList):
    def __init__(self, trial):
        super(NeuralNet, self).__init__()

        # Where the layers and dropouts will be stacked
        self.layers = []
        self.dropouts = []

        # Given a trial, takes the suggested number of 
        # layers as well as the dropout value  
        n_layers = trial.suggest_int('n_layers', 1, 4)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

        # Since the input tensor has a shape (n, 28*28)
        input_dim = 28 * 28

        # Given the suggested number of layers, proceeds by
        # stacking layer by layer. Likewise the number of internal
        # units is a value taken from the current trial suggestion
        for layer in range(n_layers):
            output_dim = trial.suggest_int(f"output_dim_{layer}", 12, 128, log=True)
            
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))

            input_dim = output_dim

        # The last layer is added to the stack.
        self.layers.append(nn.Linear(input_dim, 10))
        self.dropouts.append(nn.Dropout(0))

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

        # Softmax activation function is 
        # applied to the last layer
        x = torch.log_softmax(x, dim=1)

        return x

class Model:
    def __init__(self):
        self.train_loader = None
        self.test_loader = None

    def prepare_data(self):

        # Download and save MNIST as train & test loaders
        self.train_loader = DataLoader(MNIST(os.getcwd(), 
                            train=True, 
                            download=True, 
                            transform=transforms.ToTensor()),
                        batch_size=128,
                        shuffle=True)

        self.test_loader = DataLoader(MNIST(os.getcwd(), 
                            train=False, 
                            transform=transforms.ToTensor()),
                        batch_size=128,
                        shuffle=True)


    def optimize(self, trial):
        
        # Initialize the Neural Net model with the current trial
        neural_net = NeuralNet(trial)
        
        # Define space search for training settings
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(neural_net.parameters(), lr=learning_rate)

        # Starts training phase
        for epoch in range(10):
            
            # Set model in training model
            neural_net.train()
            
            # Starts batch training
            for x_batch, y_batch in self.train_loader:
                
                # Current batch is reshaped 
                x_batch = x_batch.view(x_batch.size(0), -1)
                
                # Feed the model
                y_pred = neural_net(x_batch)
                
                # Loss calculation
                loss = F.nll_loss(y_pred, y_batch)
                
                # Gradients calculation
                loss.backward()
                
                # Gradients update
                optimizer.step()
                
                # Clean gradientes
                optimizer.zero_grad()
            
            # Starts evaluation phase
            neural_net.eval()
            with torch.no_grad():
                
                # True positive & 
                # False positive initialization
                tp_fp = 0
                
                for x_batch, y_batch in self.test_loader:
                    
                    x_batch = x_batch.view(x_batch.size(0), -1)
                    y_pred = neural_net(x_batch)

                    pred = y_pred.argmax(dim=1, keepdim=True)
                    tp_fp += pred.eq(y_batch.view_as(pred)).sum().item()

            accuracy = tp_fp / len(self.test_loader.dataset)

        # Retun accuracy since we 
        # want it to be maximized
        return accuracy


if __name__ == '__main__':

    # Initialize the model and prepare data
    model = Model()
    model.prepare_data()
    
    # Define a study for "maximization"
    study = optuna.create_study(direction="maximize")
    # Starts optimization for 50 iterations
    study.optimize(model.optimize, n_trials=50)