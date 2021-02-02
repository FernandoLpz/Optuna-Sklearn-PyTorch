# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import optuna

class DataHandler(Dataset):

	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

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

		# Since the input tensor has a shape (n, 30)
		input_dim = 28 * 28

		# Given the suggested number of layers, proceeds by
		# stacking layer by layer. Likewise the number of internal
		# units is a value taken from the current trial suggestion
		for layer in range(n_layers):
			output_dim = trial.suggest_int(f"output_dim_{layer}", 12, 128, log=True)
			
			self.layers.append(nn.Linear(input_dim, output_dim))
			self.dropouts.append(nn.Dropout(dropout))

			input_dim = output_dim

		# The last layer is added to the stack. Sice we are working
		# with binary classification, 1 output unit is enough
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

		# Last layer has sigmoid as 
		# activation funciton
		x = torch.sigmoid(x)
		x = torch.log_softmax(x, dim=1)
		# x = F.relu(x)

		# return x.squeeze()
		return x

class Model:
	def __init__(self):
		self.train_loader = None
		self.test_loader = None

	def prepare_data(self):
		# self.dataset = pd.read_csv('data/mushrooms.csv')
		# columns_to_be_encoded = self.dataset.drop(['Class'], axis=1).columns
		# x = pd.get_dummies(self.dataset.drop(['Class'], axis=1), columns=columns_to_be_encoded)
		# classes = self.dataset['Class'].unique()
		# for idx, class_name in enumerate(classes):
		# 	self.dataset['Class'] = self.dataset['Class'].replace(class_name, idx)
		# y = self.dataset['Class']

		# x = x.values
		# y = y.values

		# # Load data and split into train and test sets
		# # x, y = load_breast_cancer(return_X_y=True)
		# self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.30, random_state=12)

		# MNIST
		self.train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
		batch_size=128,
		shuffle=True)

		self.test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(os.getcwd(), train=False, transform=transforms.ToTensor()),
		batch_size=128,
		shuffle=True)


	def optimize(self, trial):
		
		# Initialize the Neural Net model with the current trial
		neural_net = NeuralNet(trial)
		
		# Define space search for training settings
		optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
		learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
		optimizer = getattr(optim, optimizer_name)(neural_net.parameters(), lr=learning_rate)

		# # Initialize data handler
		# train = DataHandler(self.x_train, self.y_train)
		# test = DataHandler(self.x_test, self.y_test)

		# # Initialize dataset loader
		# self.train_loader = DataLoader(train, batch_size=batch_size)
		# self.test_loader = DataLoader(test, batch_size=batch_size)


		# Starts training phase
		for epoch in range(10):
			# Set model in training model
			neural_net.train()
			
			# Starts batch training
			for x_batch, y_batch in self.train_loader:

				# x_batch = x_batch.type(torch.FloatTensor)
				# y_batch = y_batch.type(torch.FloatTensor)
				x_batch = x_batch.view(x_batch.size(0), -1)
				# Clean gradientes
				optimizer.zero_grad()
				
				# Feed the model
				y_pred = neural_net(x_batch)
				
				# Loss calculation
				loss = F.nll_loss(y_pred, y_batch)
				
				
				# Gradients calculation
				loss.backward()
				
				# Gradients update
				optimizer.step()
			
			neural_net.eval()
			tp, fp = 0, 0
			correct = 0
			with torch.no_grad():
				for x_batch, y_batch in self.test_loader:

					# x_batch = x_batch.type(torch.FloatTensor)
					# y_batch = y_batch.type(torch.FloatTensor)
					x_batch = x_batch.view(x_batch.size(0), -1)
					y_pred = neural_net(x_batch)

					pred = y_pred.argmax(dim=1, keepdim=True)
					correct += pred.eq(y_batch.view_as(pred)).sum().item()

					# for pred, true in zip(y_pred, y_batch):
					# 	if (pred >= 0.5) and (true == 1):
					# 		tp += 1
					# 	if (pred < 0.5) and (true == 0):
					# 		fp += 1
			accuracy = correct / len(self.test_loader.dataset)
				# accuracy = (tp + fp) / len(self.y_test)
	
		return accuracy


if __name__ == '__main__':

	model = Model()
	model.prepare_data()
	
	study = optuna.create_study(direction="maximize")
	study.optimize(model.optimize, n_trials=50)