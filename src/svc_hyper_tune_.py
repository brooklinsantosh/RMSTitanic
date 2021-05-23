import pandas as pd
from sklearn.svm import SVC
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("../input/train_cleaned_folds.csv")
    X = df.drop(["Survived", "kfold"], axis=1)
    y = df["Survived"]

    clf = SVC(random_state=42)
    param_grid = {
        'gamma' : [0.0001, 0.001, 0.01, 0.1],
        'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    }

    model = model_selection.GridSearchCV(
        estimator= clf,
        param_grid= param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    
    )
    model.fit(X,y)
    print("Best model values")
    print(model.best_score_)
    print(model.best_estimator_)
