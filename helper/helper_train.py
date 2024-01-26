# Statistics and visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Machine learning
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Deep learning
from keras.models import Model
from keras import Sequential
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Dense, Activation, Add, Flatten)

# Metrics
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
    plt.figure(figsize=(10, 5.5))

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


# CNN
def ResNet1D(input_dims, residual_blocks_dims, nclasses,
             dropout_rate=0.8, kernel_size=2, kernel_initializer='he_normal'):
    # Residual Block
    def residual_block(X, nsamples_in, nsamples_out,
                       nfilters_in, nfilters_out, first_block):
        # Skip Connection
        # Deal with downsampling
        downsample = nsamples_in // nsamples_out

        if downsample > 1:
            Y = MaxPooling1D(downsample, strides=downsample, padding='same')(X)
        elif downsample == 1:
            Y = X
        else:
            raise ValueError("Number of samples should always decrease.")
        
        # Deal with nfiters dimension increase
        if nfilters_in != nfilters_out:
            Y = Conv1D(nfilters_out, 1, padding='same', 
                       use_bias=False, kernel_initializer=kernel_initializer)(Y)

        # End of 2nd layer from last residual block
        if not first_block:
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
            X = Dropout(dropout_rate)(X)

        # 1st layer
        X = Conv1D(nfilters_out, kernel_size, strides=downsample, padding='same', 
                   use_bias=False, kernel_initializer=kernel_initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(dropout_rate)(X)

        # 2nd layer
        X = Conv1D(nfilters_out, kernel_size, padding='same',
                   use_bias=False, kernel_initializer=kernel_initializer)(X)
        X = Add()([X, Y]) # Skip connection and main connection
        return X
    
    # Define input representing IMU_data
    IMU_data = Input(shape=input_dims, dtype=np.float32, name='IMU_data')
    X = IMU_data
    
    # First layer
    downsample = input_dims[0] // residual_blocks_dims[0][0]
    X = Conv1D(residual_blocks_dims[0][1], kernel_size, strides=downsample, 
               padding='same', kernel_initializer=kernel_initializer, 
               use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Residual blocks
    first_block = True
    nresidual_blocks = len(residual_blocks_dims) - 1
    for i in range(nresidual_blocks):
        X = residual_block(X, residual_blocks_dims[i][0], residual_blocks_dims[i+1][0], 
                           residual_blocks_dims[i][1], residual_blocks_dims[i+1][1], first_block)
        first_block = False   
        
    # End of 2nd layer from last residual block
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(dropout_rate)(X)

    # Last layer
    X = Flatten()(X)
    diagnosis = Dense(nclasses, kernel_initializer=kernel_initializer, 
                      activation='softmax', name='diagnosis')(X)

    return Model(IMU_data, diagnosis)


def CNN(input_dims, nclasses, kernel_size=2):
    n_features = input_dims[0]

    # Init model
    model = Sequential()

    # Add layers
    model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu', input_shape=(n_features, 1), padding='same'))
    model.add(Flatten())   # 1D array
    model.add(Dense(nclasses, activation='softmax'))    
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model


# Comparison model
class ModelToCompare():
    def __init__(self, sensor, accuracy, model, X_train, X_test, y_train, y_test):
        self.sensor = sensor
        self.accuracy = accuracy
        self.model = model