import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statistics import mode


df = pd.read_csv("../input/train_cleaned_folds.csv")
test = pd.read_csv("../input/test_cleaned.csv")
ss = pd.read_csv("../input/gender_submission.csv")

clf = RandomForestClassifier(
    n_jobs=-1, 
    n_estimators=364, 
    criterion="entropy", 
    max_depth=12,
    max_features= 0.9578120305865521,
    bootstrap= "True",
    min_samples_leaf= 1,
    min_samples_split= 8)

accuracy =[]
preds = []
for f in range(5):
    train = df[df.kfold!= f].reset_index(drop=True)
    valid = df[df.kfold== f].reset_index(drop=True)

    X_train = train.drop(["Survived", "kfold"], axis=1)
    y_train = train["Survived"]
    X_valid = valid.drop(["Survived", "kfold"], axis=1)
    y_valid = valid["Survived"]
    X_test = test.drop("Survived", axis=1)

    clf.fit(X_train,y_train)
    valid_preds = clf.predict(X_valid)
    preds.append(clf.predict(X_test))

    accuracy.append(accuracy_score(y_valid,valid_preds))

print(accuracy)
print(sum(accuracy)/5)

rf_pred = np.array([])
for i in range(0,len(X_test)):
    rf_pred = np.append(rf_pred, mode([preds[0][i],preds[1][i],preds[2][i],preds[3][i],preds[4][i]]))

ss["Survived"] = rf_pred
ss.to_csv("../predictions/rf_pred.csv")