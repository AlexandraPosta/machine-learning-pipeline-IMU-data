import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.inspection import permutation_importance


label_map = {
    1: 'LGW',
    2: 'Ramp_ascend',
    3: 'Ramp_descend',
    4: 'Sit_to_stand',
    5: 'Stand_to_sit',
}


# Metrics
def get_metrics(true, pred):
    print("Accuracy: ", metrics.accuracy_score(true, pred))
    print("Precision: ", metrics.precision_score(true, pred, average='weighted'))
    print("Recall: ", metrics.recall_score(true, pred, average='weighted', zero_division=0))
    print("Classification error: ", 1 - metrics.accuracy_score(true, pred))
    print(metrics.classification_report(true, pred, zero_division=0))


def plot_cm(y_true, y_pred, labels, font_scale=0.8): 
    labels = [1, 2, 3, 4, 5]
    # Plot the confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, normalize='true')    
    fig, ax = plt.subplots() 
    ax = sns.heatmap(cm, annot=True, linewidth=0.01, fmt=".3f", ax=ax)

    plt.ylabel('Actual', fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.set_yticklabels(labels, fontsize=12, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.show()


def plot_val(grid):
    cv_results = grid.cv_results_
    num_folds = grid.cv

    # Plotting the results
    plt.figure(figsize=(10, 6))

    for i in range(len(cv_results['params'])):
        fold_accuracies = [cv_results[f'split{j}_test_score'][i] for j in range(num_folds)]
        plt.plot(fold_accuracies, label=f"Param set {i+1}")

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Fold-wise Accuracy for Each Parameter Set')
    plt.legend()
    plt.show()

# ANN model
class ANN():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train.values.ravel()
        self.X_test  = X_test
        self.y_test  = y_test.values.ravel()
        self.model = MLPClassifier()

    def run_pipeline(self):
        self.get_hyperparameter_tuning()
        self.train()
        self.predict()

    def train(self):
        self.model = MLPClassifier(**self.best_params, random_state=10)

        # Train model
        self.model.fit(self.X_train, self.y_train)
        self.coef = self.model.coefs_[0]

    def get_hyperparameter_tuning(self):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100), (100, 100)],
            'activation': ['relu','tanh','logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant','adaptive'],
            'solver': ['lbfgs'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 300, 400],
            'batch_size': [16, 32, 64, 128],
            'early_stopping': [True, False]
        }
        
        """
        # Appy scalar
        scaler = StandardScaler() 
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        """
        
        # Grid search random
        estimator = MLPClassifier()
        gsc_random = RandomizedSearchCV(estimator, param_grid, cv=5, verbose=-1, random_state=42, n_jobs=-1)
        gsc_random.fit(self.X_train, self.y_train)
        self.grid_search = gsc_random
        self.best_params =  gsc_random.best_params_

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        pred = [round(item) for item in self.y_pred]
        get_metrics(self.y_test, pred)
        label = [label_map[i] for i in self.model.classes_]
        plot_cm(self.y_test, pred, label)
        plot_val(self.grid_search)

    def get_accuracy(self):
        pred = [round(item) for item in self.y_pred]
        return metrics.accuracy_score(self.y_test, pred)
    
    def get_classification_error(self):
        pred = [round(item) for item in self.y_pred]
        return 1-metrics.accuracy_score(self.y_test, pred)

    def get_most_relevant_features(self, columns=None, X=None, y=None):
        perm_importance = permutation_importance(self.model, 
                                                 X, 
                                                 y, 
                                                 random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()

        # Return features names and their importance score
        return columns[sorted_idx[-15:]], perm_importance.importances_mean[sorted_idx[-15:]]


# SVM model
class SVM():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train.values.ravel()
        self.X_test  = X_test
        self.y_test  = y_test.values.ravel()
        self.model = SVC()

    def run_pipeline(self):
        self.get_hyperparameter_tuning()
        self.train()
        self.predict()

    def train(self):
        self.model = SVC(**self.best_params, random_state=10)

        # Train model
        self.model.fit(self.X_train, self.y_train)
        self.coef = self.model.coef_

    def get_hyperparameter_tuning(self):
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear'] # otherwise we cannot access coeficients
        }

        # Appy scalar
        """	
        scaler = StandardScaler() 
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        """

        # Grid search random
        estimator = SVC()
        gsc_random = RandomizedSearchCV(estimator, param_grid, cv=5, verbose=-1, random_state=42, n_jobs=-1)
        gsc_random.fit(self.X_train, self.y_train)
        self.grid_search = gsc_random
        self.best_params =  gsc_random.best_params_

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        pred = [round(item) for item in self.y_pred]
        get_metrics(self.y_test, pred)
        label = [label_map[i] for i in self.model.classes_]
        plot_cm(self.y_test, pred, label)
        plot_val(self.grid_search)

    def get_accuracy(self):
        pred = [round(item) for item in self.y_pred]
        return metrics.accuracy_score(self.y_test, pred)
    
    def get_classification_error(self):
        pred = [round(item) for item in self.y_pred]
        return 1-metrics.accuracy_score(self.y_test, pred)

    def get_most_relevant_features(self, columns=None, X=None, y=None):
        perm_importance = permutation_importance(self.model, 
                                                 X, 
                                                 y, 
                                                 random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()

        # Return features names and their importance score
        return columns[sorted_idx[-15:]], perm_importance.importances_mean[sorted_idx[-15:]]


# Comparison model
class ModelToCompare():
    def __init__(self, sensor, accuracy, model, X_train, X_test, y_train, y_test):
        self.sensor = sensor
        self.accuracy = accuracy
        self.model = model