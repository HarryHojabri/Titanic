The Titanic project, taken from Kaggle, is about creating a model to predict which passengers survived the Titanic shipwreck:

https://www.kaggle.com/c/titanic/

There are 2 datasets in the project:

- "train.csv" which is the labeled training set including different information such as gender and age for 891 passengers and their labels, "Survived" indicating if a passenger survived or not.

- "test.csv" including 419 unlabeled passengers: After training the model on train.csv, the labels are calculated for the this dataset and stored in the corresponding .csv file.

The library scikit-learn is imported and different models such as Logistic Regression and Random Forest are applied to solve the problem.
