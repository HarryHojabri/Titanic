import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def clean_data(datafile):


    datafile["Fare"] = datafile["Fare"].fillna(datafile["Fare"].dropna().median())
    datafile["Age"] = datafile["Age"].fillna(datafile["Age"].dropna().median())
    datafile["Embarked"] = datafile["Embarked"].fillna("S")

    datafile["Age"] = pd.get_dummies(datafile["Age"])
    datafile["Pclass"] = pd.get_dummies(datafile["Pclass"])
    datafile["Sex"] = pd.get_dummies(datafile["Sex"])
    datafile["SibSp"] = pd.get_dummies(datafile["SibSp"])
    datafile["Parch"] = pd.get_dummies(datafile["Parch"])
    datafile["Embarked"] = pd.get_dummies(datafile["Embarked"])
