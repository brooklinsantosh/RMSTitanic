import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from functools import partial
import optuna
from sklearn.preprocessing import MinMaxScaler

def optimize(trial, x, y):
    max_iter = trial.suggest_int("max_iter", 100, 1000)
    C = trial.suggest_uniform("C", 0.01, 20)
    solver = trial.suggest_categorical("solver", ["newton-cg", "lbfgs","liblinear","sag","saga"])


    model = LogisticRegression(
        max_iter= max_iter,
        C= C,
        solver= solver,
        n_jobs=-1,
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

    #One Hot Encode the categorical column
    df=pd.get_dummies(data=df,drop_first=True,columns=['Pclass', 'SibSp', 'Parch', 'Embarked', 'Title'])
    #Scale the numerical data
    scl = MinMaxScaler()
    scaled = scl.fit_transform(df[["Age","Fare"]])
    df.loc[:,["Age","Fare"]]=scaled

    X = df.drop(["Survived", "kfold"], axis=1)
    y = df["Survived"]

    optimization_funtion =partial(optimize, x=X, y=y)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_funtion, n_trials=15)
