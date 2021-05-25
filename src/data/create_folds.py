import pandas as pd
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("../input/train_cleaned.csv")

train["kfold"] = -1
train = train.sample(frac=1).reset_index(drop=True)
y = train["Survived"]
kf = StratifiedKFold(n_splits=5)
for f, (t_,v_) in enumerate(kf.split(X=train,y=y)):
  train.loc[v_,"kfold"] = f

train.to_csv("../input/train_cleaned_folds.csv",index=False)