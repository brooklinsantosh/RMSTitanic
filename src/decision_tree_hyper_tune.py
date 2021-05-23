import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from functools import partial
import optuna

def optimize(trial, x, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 3 ,15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1 ,5)
    min_samples_split = trial.suggest_int("min_samples_split", 2 ,10)



    model = DecisionTreeClassifier(
        max_depth= max_depth,
        max_features= max_features,
        criterion= criterion,
        splitter= splitter,
        min_samples_leaf= min_samples_leaf,
        min_samples_split = min_samples_split,
        random_state=42
    )
    accuracies = []
    for f in range(5):
        train = df[df.kfold!= f].reset_index(drop=True)
        valid = df[df.kfold== f].reset_index(drop=True)
        xtrain = train.drop(["Survived", "kfold"], axis=1)
        ytrain = train["Survived"]
        xvalid = valid.drop(["Survived", "kfold"], axis=1)
        yvalid = valid["Survived"]

        model.fit(xtrain,ytrain)
        preds = model.predict(xvalid)
        fold_acc = accuracy_score(yvalid, preds)
        accuracies.append(fold_acc)
    
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/train_cleaned_folds.csv")
    X = df.drop(["Survived", "kfold"], axis=1)
    y = df["Survived"]

    optimization_funtion =partial(optimize, x=X, y=y)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_funtion, n_trials=15)
