#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import glob
import pandas as pd
import math
import glob
import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy import matrix 
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,  precision_score, recall_score, classification_report, confusion_matrix
import sklearn.metrics
import multiprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(model, param_grid, X, y):
    skf = StratifiedKFold(n_splits=10)
    recall = []
    precision = []
    f = []
    accuracy = []
    roc = []

    for train_index, test_index in skf.split(X, y):
        xtr, xvl = X.loc[train_index], X.loc[test_index]
        ytr, yvl = y.loc[train_index], y.loc[test_index]

        ytr = ytr.to_numpy().reshape(len(ytr),)
        clf = GridSearchCV(model, param_grid, scoring='roc_auc')
        clf.fit(xtr, ytr)
        clf_best = clf.best_estimator_

        y_pred = clf.best_estimator_.predict(xvl)

        yvl = yvl.to_numpy().reshape(len(yvl),)
        n_classes = len(np.unique(yvl))
        yvl1 = label_binarize(yvl, classes=np.arange(n_classes))
        y_pred1 = label_binarize(y_pred, classes=np.arange(n_classes))

        roc.append(roc_auc_score(yvl,y_pred))
        accuracy.append(accuracy_score(yvl, y_pred))
        recall.append(recall_score(yvl,y_pred))
        precision.append(precision_score(yvl,y_pred,average='weighted'))
      
        f.append(sklearn.metrics.f1_score(yvl,y_pred, average='weighted'))

    return (np.mean(roc), np.std(roc), np.mean(accuracy), np.std(accuracy),
            np.mean(recall), np.std(recall), np.mean(precision), np.std(precision), clf_best)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def parallel_evaluate(model_name, X_train, y_train):
    if model_name == 'RF':
        param_grid = {
            'n_estimators': [1, 2, 3, 5, 10, 30, 50, 100, 200, 300, 500]
        }
        model = RandomForestClassifier(random_state=1234)
    elif model_name == 'SVM':
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        }
        model = SVC(probability=True, random_state=1234)
    elif model_name == 'NB':
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=100)
        }
        model = GaussianNB()
    elif model_name == 'MLP':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive']
        }
        model = MLPClassifier(random_state=1234)
    elif model_name == 'KNN':
        param_grid = {
            'n_neighbors': [1, 10],  # Corrigido para refletir um intervalo ou lista de valores
            'leaf_size': [20, 40],  # Corrigido para refletir uma lista de valores
            'p': [1, 2],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'chebyshev']
        }
        model = KNeighborsClassifier()
    elif model_name == 'LR':
        param_grid = {
            "C": np.logspace(-3, 3, 7),
            "penalty": ["l1", "l2"]
        }
        model = LogisticRegression(random_state=1234, solver='liblinear')  # Ajustado para suportar L1

    # Avaliação do modelo (implementação da função evaluate_model não fornecida)
    roc_mean, std_roc, accuracy_mean, std_accuracy, recall_mean, std_recall, precision_mean, std_precision, clf = evaluate_model(model, param_grid, X_train, y_train)

    return {
        'Model': model_name,
        'AUC': roc_mean,
        'AUC std': std_roc,
        'Accuracy': accuracy_mean,
        'Accuracy std': std_accuracy,
        'Recall': recall_mean,
        'Recall std': std_recall,
        'Precision': precision_mean,
        'Precision std': std_precision,
        'clf_best': clf
    }
