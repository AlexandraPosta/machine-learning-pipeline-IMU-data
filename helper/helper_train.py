from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Metrics
def get_metrics(true, pred):
    pred = [round(item) for item in pred]
    print("Accuracy: ", metrics.accuracy_score(true, pred))
    print("Precision: ", metrics.precision_score(true, pred, average='weighted'))
    print("Recall: ", metrics.recall_score(true, pred, average='weighted', zero_division=0))
    print(classification_report(true, pred, zero_division=0))
    print(confusion_matrix(true, pred))


# ANN model


# SVM model


# CNN model