import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import optuna

class ModelOptimization:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def optimize(self, trial):
        
        # Definition of space search
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 2, 10)

        # Classifier definition
        model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    criterion=criterion)

        avg_accuracy = []

        # Definition of k-fold cross validation
        k_fold = KFold(n_splits=5)

        for train_idx, test_idx in k_fold.split(x, y):
            
            # Training fold
            x_train = x[train_idx]
            y_train = y[train_idx]
            
            # Testing fold
            x_test = x[test_idx]
            y_test = y[test_idx]

            # Training
            model.fit(x_train, y_train)

            # Save accuracy
            avg_accuracy.append(model.score(x_test, y_test))
        
        return np.mean(avg_accuracy)
    
    def train_optimal_configuration(self, best_params):

        # Classifier definition with optimal paramters
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                    max_depth=best_params['max_depth'],
                                    criterion=best_params['criterion'])

        return model.fit(self.x, self.y)

    
if __name__ == '__main__':

    # Load data and split into train and test sets
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=23)

    # Initialize the optimization model class
    modelOpt = ModelOptimization(x=x_train, y=y_train)

    # Study initialization
    # direction = 'maximize' : since the goal is to maximize the accuracy
    # sampler = 'TPEsampler' : default paramter for single-objective optimization
    # pruner = 'MedianPruner' : default paramter for pruning useless configurations
    study = optuna.create_study(direction='maximize')

    # Opimization
    # Recevies the function to be optimized and the number of trials
    study.optimize(modelOpt.optimize, n_trials=50)

    # Since we aimed to maximize accuracy, prints the best 
    # accuracy obtained as well as the best configuration
    print(f"Best average accuracy: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # Trains a model with the optimal parameters
    optimal_model = modelOpt.train_optimal_configuration(study.best_params)

    # Print final results
    print(f"Train score: {optimal_model.score(x_train, y_train)}")
    print(f"Test score: {optimal_model.score(x_test, y_test)}")
