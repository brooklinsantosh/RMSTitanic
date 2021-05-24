import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from create_result_df import CreateResult
from statistics import mode

df = pd.read_csv("../input/train_cleaned_folds.csv")
test = pd.read_csv("../input/test_cleaned.csv")
ss = pd.read_csv("../input/gender_submission.csv")

#Change the Outliers which are only present in test data to the max value
test.loc[[342,365],"Parch"] = 6
#One Hot Encode the categorical column
df=pd.get_dummies(data=df,drop_first=True,columns=['Pclass', 'SibSp', 'Parch', 'Embarked', 'Title'])
test=pd.get_dummies(data=test,drop_first=True,columns=['Pclass', 'SibSp', 'Parch', 'Embarked', 'Title'])
#Scale the numerical data
scl = MinMaxScaler()
scaled = scl.fit_transform(df[["Age","Fare"]])
scaled_test = scl.transform(test[["Age","Fare"]])
df.loc[:,["Age","Fare"]]=scaled
test.loc[:,["Age","Fare"]]=scaled_test

clf = LogisticRegression(
    max_iter=839,
    C=2.679113846370858,
    solver="lbfgs",
    n_jobs=-1,
    random_state=42)

# clf = LogisticRegression(
#     max_iter=904,
#     C=2.63197600686173,
#     solver="saga",
#     n_jobs=-1,
#     random_state=42)

accuracy =[]
preds = []
for f in range(5):
    train = df[df.kfold!= f].reset_index(drop=True)
    valid = df[df.kfold== f].reset_index(drop=True)

    X_train = train.drop(["Survived", "kfold"], axis=1)
    y_train = train["Survived"]
    X_valid = valid.drop(["Survived", "kfold"], axis=1)
    y_valid = valid["Survived"]
    X_test = test.drop(["Survived"], axis=1)

    clf.fit(X_train,y_train)
    valid_preds = clf.predict(X_valid)
    preds.append(clf.predict(X_test))

    accuracy.append(accuracy_score(y_valid,valid_preds))

print(accuracy)
print(sum(accuracy)/5)

logreg_pred = np.array([])
for i in range(0,len(X_test)):
    logreg_pred = np.append(logreg_pred, mode([preds[0][i],preds[1][i],preds[2][i],preds[3][i],preds[4][i]]))

#Add these data to result dataframe
c = CreateResult(accuracy,"LogisticRegression")
c.create()

ss["Survived"] = logreg_pred
ss["Survived"] = ss["Survived"].astype(int)
ss.to_csv("../predictions/logreg_pred.csv",index=False)