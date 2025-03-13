import numpy as np
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys
import pandas as pd

from DNNPredict import DNNPredict
from CNNPredict import CNNPredict

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=1),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=1),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=1)
    }

def soft_voting(resps):
    y_proba_avg = np.mean([resp["y_proba"] for resp in resps], axis=0)
    y_pred = np.argmax(y_proba_avg, axis=1)
    return compute_metrics(resps[0]["y_true"], y_pred)

def stacking_decision_tree(X, y_true, X_test, y_true_test):
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y_true)
                                                                                                               
    y_pred = model.predict(X_test)
    return compute_metrics(y_true_test, y_pred)

def bagging_ensemble(X, y_true, X_test, y_true_test):
   
    model = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=42)
    model.fit(X, y_true)
    
    y_pred = model.predict(X_test)
    return compute_metrics(y_true_test, y_pred)

def gradient_boosting(X, y_true, X_test, y_true_test):   
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    model.fit(X, y_true)
    
    
    y_pred = model.predict(X_test)
    return compute_metrics(y_true_test, y_pred)

def stacking_mlp(X, y_true, X_test, y_true_test):
    
    input_dim = X.shape[1]
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y_true)), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_true, epochs=30, batch_size=32, verbose=0)
    

    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return compute_metrics(y_true_test, y_pred)

def run_ensemble_methods(resps, resps_val2):
    results = []
    #create X and y_true based on val2
#    X = np.hstack([resp["y_proba"] for resp in resps_val2])
#    y_true = resps_val2[0]["y_true"].values.ravel() if hasattr(resps_val2[0]["y_true"], 'values') else np.array(resps_val2[0]["y_true"]).ravel()

    #create X and y_true based on test dataset for meta model evaluation
    X_test = np.hstack([resp["y_proba"] for resp in resps])
    y_true_test = resps[0]["y_true"].values.ravel() if isinstance(resps[0]["y_true"], (pd.DataFrame, pd.Series)) else np.array(resps[0]["y_true"]).ravel()

    results.append({"ensemble_method": "Soft Voting", "ensemble_metrics": soft_voting(resps)})
    
    
#    results.append({"ensemble_method": "Stacking - Decision Tree", "ensemble_metrics": stacking_decision_tree(X, y_true, X_test, y_true_test)})
#    results.append({"ensemble_method": "Bagging", "ensemble_metrics": bagging_ensemble(X, y_true, X_test, y_true_test)})
#    results.append({"ensemble_method": "Gradient Boosting", "ensemble_metrics": gradient_boosting(X, y_true, X_test, y_true_test)})
#    results.append({"ensemble_method": "Stacking - MLP", "ensemble_metrics": stacking_mlp(X, y_true, X_test, y_true_test)})

    return results

def run_ensemble(model_files, DNN_test, CNN_test, DNN_val2, CNN_val2):
    resps = []
    resps_val2 = []
    for file in model_files:
        if os.path.splitext(file)[1] == ".keras":
            resps.append(DNNPredict(file, DNN_test, "emotion"))
#            resps_val2.append(DNNPredict(file, DNN_val2, "emotion"))
        elif os.path.splitext(file)[1] == ".pth":
            resps.append(CNNPredict(file, CNN_test, "emotion"))
#            resps_val2.append(CNNPredict(file, CNN_val2, "emotion"))
    
    features = [n['features'] for n in resps]
    ind_accuracy = [n['accuracy'] for n in resps]
    return run_ensemble_methods(resps, resps_val2), features, ind_accuracy


