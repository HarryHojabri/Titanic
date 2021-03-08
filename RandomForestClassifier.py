import numpy as np
import pandas as pd
import data_cleanning


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


from sklearn.ensemble import RandomForestClassifier
import csv

label_train = train_data["Survived"]

data_cleanning.clean_data(train_data)
data_cleanning.clean_data(test_data)

features = ["Age", "Pclass", "Sex", "SibSp", "Parch"]
x_train = train_data[features]
x_test = test_data[features]



rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rfc.fit(x_train, label_train)
predictions = rfc.predict(x_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('random_forest_submission1.csv', index=False)
print("Your submission was successfully saved!")
