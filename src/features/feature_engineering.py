import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def get_title(st):
  return st.split(",")[1].split(".")[0]

def alone(sp):
  if (sp[0]+sp[1]==0):
    return 1
  else:
     return 0

def check_adult_male(a):
  if a[0]>=18:
    if a[1]=="male":
      return 1
    else:
      return 0
  else:
    return 0

test["Survived"] = -1
df = pd.concat([train,test], axis=0)

#Extracting title from Name
df["Title"] = df.Name.apply(get_title)
df["Title"]=df["Title"].replace([" Don"," Dona"," Rev"," Dr"," Major"," Lady"," Sir"," Col"," Capt"," the Countess"," Jonkheer"]," Rare")
df["Title"]=df["Title"].replace([" Mlle", " Ms"]," Miss")
df["Title"]=df["Title"].replace([" Mme"]," Mrs")

#Creating new columns
df["Alone"] = df[["SibSp","Parch"]].apply(alone,axis=1)
df["AdultMale"] = df[["Age","Sex"]].apply(check_adult_male,axis=1)

#Fill missing values
for t in df["Title"].unique().tolist():
  m = df[df["Title"]== t]["Age"].mean()
  mask = df["Title"]== t
  df.loc[mask,"Age"] = df.loc[mask,"Age"].fillna(m)

df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

#Dropping unneccesary columns
df.drop(["Name","PassengerId", "Ticket"], axis=1, inplace=True)
df.drop("Cabin", axis=1, inplace=True) #Because more NAN values

#Label encoding the data
sex_map = {"male":0, "female":1}
embarked_map = {"S":0, "C":1, "Q":2}
title_map = {" Mr":0," Miss":1," Mrs":2," Master":3," Rare":4}

df["Sex"] = df["Sex"].map(sex_map)
df["Embarked"] = df["Embarked"].map(embarked_map)
df["Title"] = df["Title"].map(title_map)

#Type casting the columns to it correct type
df["Survived"] = df["Survived"].astype("category")
df["Pclass"] = df["Pclass"].astype("category")
df["Sex"] = df["Sex"].astype("category")
df["SibSp"] = df["SibSp"].astype("category")
df["Parch"] = df["Parch"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")
df["Title"] = df["Title"].astype("category")

# print(df.head(20))
# print(df.isnull().sum())

#Splitting the dataframes
train = df[df["Survived"] != -1].reset_index(drop=True)
test = df[df["Survived"] == -1].reset_index(drop=True)

# Writing the dataframe
train.to_csv("../input/train_cleaned.csv",index=False)
test.to_csv("../input/test_cleaned.csv",index=False)

