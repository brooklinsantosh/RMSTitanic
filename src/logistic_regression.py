import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("../input/train_cleaned_folds.csv")
test = pd.read_csv("../input/test_cleaned.csv")
ss = pd.read_csv("../input/gender_submission.csv")

clf = LogisticRegression(n_jobs=-1)

accuracy =[]
preds = []
for f in range(5):
    train = df[df.kfold!= f].reset_index(drop=True)
    valid = df[df.kfold== f].reset_index(drop=True)

    X_train = train.drop(["Survived", "kfold", "Age", "Fare", "Alone","AdultMale"], axis=1)
    y_train = train["Survived"]
    X_valid = valid.drop(["Survived", "kfold", "Age", "Fare", "Alone","AdultMale"], axis=1)
    y_valid = valid["Survived"]
    X_test = test.drop(["Survived","Age", "Fare", "Alone","AdultMale"], axis=1)

    print(X_train.head())

    clf.fit(X_train,y_train)
    valid_preds = clf.predict(X_valid)
    preds.append(clf.predict(X_test))

    accuracy.append(accuracy_score(y_valid,valid_preds))

print(accuracy)
print(sum(accuracy)/5)