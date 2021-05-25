import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from functools import partial
import optuna

def optimize(trial, x, y):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"])
    gamma = trial.suggest_uniform("gamma", 0.0001, 0.1)
    C = trial.suggest_uniform("C", 0.01, 20)


    model = SVC(
        gamma=gamma,
        C=C,
        kernel=kernel,
        random_state= 42
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
    study.optimize(optimization_funtion, n_trials=25)
