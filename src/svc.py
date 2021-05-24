import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from statistics import mode
from create_result_df import CreateResult

df = pd.read_csv("../input/train_cleaned_folds.csv")
test = pd.read_csv("../input/test_cleaned.csv")
ss = pd.read_csv("../input/gender_submission.csv")

#clf = SVC(C=0.05, gamma=0.0001, kernel='linear', random_state=42)

clf = SVC(C=0.07259632547748618, gamma=0.06186828707791769, kernel='linear', random_state=42)

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

svc_pred = np.array([])
for i in range(0,len(X_test)):
    svc_pred = np.append(svc_pred, mode([preds[0][i],preds[1][i],preds[2][i],preds[3][i],preds[4][i]]))

#Add these data to result dataframe
c = CreateResult(accuracy,"SVC")
c.create()

ss["Survived"] = svc_pred
ss["Survived"] = ss["Survived"].astype(int)
ss.to_csv("../predictions/svc_pred.csv",index=False)