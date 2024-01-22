import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.inspection import permutation_importance

# Metrics
def get_metrics(true, pred):
    pred = [round(item) for item in pred]
    print("Accuracy: ", metrics.accuracy_score(true, pred))
    print("Precision: ", metrics.precision_score(true, pred, average='weighted'))
    print("Recall: ", metrics.recall_score(true, pred, average='weighted', zero_division=0))
    print(metrics.classification_report(true, pred, zero_division=0))
    print(metrics.confusion_matrix(true, pred))
    print("Classification error: ", 1/metrics.accuracy_score(true, pred))


# ANN model
class ANN():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.model = MLPRegressor()

    def run_pipeline(self):
        self.get_hyperparameter_tuning()
        self.train()
        self.predict()

    def train(self):
        self.model = MLPRegressor(**self.best_params, random_state=10)

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
            'max_iter': [100, 200, 400, 800, 1000],
            'batch_size': [16, 32, 64, 128],
            'early_stopping': [True, False]
        }

        # Appy scalar
        scaler = StandardScaler() 
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Grid search random
        estimator = MLPRegressor()
        gsc_random = RandomizedSearchCV(estimator, param_grid, cv=5, verbose=-1, random_state=42, n_jobs=-1)
        gsc_random.fit(self.X_train, self.y_train)
        self.best_params =  gsc_random.best_params_

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        self.evaluate()

    def evaluate(self):
        #get_metrics(self.y_test, self.y_pred)
        pass

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
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
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
            'gamma': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear'] # otherwise we cannot access coeficients
        }

        # Appy scalar
        scaler = StandardScaler() 
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Grid search random
        estimator = SVC()
        gsc_random = RandomizedSearchCV(estimator, param_grid, cv=5, verbose=-1, random_state=42, n_jobs=-1)
        gsc_random.fit(self.X_train, self.y_train)
        self.best_params =  gsc_random.best_params_

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        self.evaluate()

    def evaluate(self):
        #get_metrics(self.y_test, self.y_pred)
        pass

    def get_most_relevant_features(self, columns=None, X=None, y=None):
        perm_importance = permutation_importance(self.model, 
                                                 X, 
                                                 y, 
                                                 random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()

        # Return features names and their importance score
        return columns[sorted_idx[-15:]], perm_importance.importances_mean[sorted_idx[-15:]]


# CNN model