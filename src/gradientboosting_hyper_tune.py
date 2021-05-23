import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from functools import partial
import optuna

def optimize(trial, x, y):
    criterion = trial.suggest_categorical("criterion", ["friedman_mse", "mse"])
    n_estimatiors = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth", 3 ,15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    learning_rate = trial.suggest_uniform("learning_rate", 0.001, 1.0)
    loss = trial.suggest_categorical("loss", ["deviance", "exponential"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1 ,10)
    min_samples_split = trial.suggest_int("min_samples_split", 2 ,10)



    model = GradientBoostingClassifier(
        n_estimators= n_estimatiors,
        max_depth= max_depth,
        max_features= max_features,
        criterion= criterion,
        learning_rate= learning_rate,
        loss= loss,
        min_samples_leaf= min_samples_leaf,
        min_samples_split = min_samples_split,
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
    study.optimize(optimization_funtion, n_trials=20)
