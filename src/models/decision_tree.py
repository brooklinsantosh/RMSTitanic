import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from statistics import mode
from create_result_df import CreateResult

df = pd.read_csv("../input/train_cleaned_folds.csv")
test = pd.read_csv("../input/test_cleaned.csv")
ss = pd.read_csv("../input/gender_submission.csv")

clf = DecisionTreeClassifier(
    criterion="gini", 
    max_depth=13,
    max_features=0.9436127289449213,
    splitter="random",
    min_samples_leaf=4,
    min_samples_split=9,
    random_state=42
    )

# clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 17, min_samples_leaf = 9,
#                              min_samples_split = 9, splitter = 'random')

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

    #print(clf.score(X_valid,y_valid))

    accuracy.append(accuracy_score(y_valid,valid_preds))

print(accuracy)
print(sum(accuracy)/5)

dt_pred = np.array([])
for i in range(0,len(X_test)):
    dt_pred = np.append(dt_pred, mode([preds[0][i],preds[1][i],preds[2][i],preds[3][i],preds[4][i]]))

#Add these data to result dataframe
c = CreateResult(accuracy,"DecisionTreeClassifier")
c.create()

ss["Survived"] = dt_pred
ss["Survived"] = ss["Survived"].astype(int)
ss.to_csv("../predictions/dt_pred.csv", index=False)