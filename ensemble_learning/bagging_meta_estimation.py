# importing required libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# read the train and test dataset
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')

# Now, we have used a dataset which has more categorical variables
# hr-employee attrition data where target variable is Attrition 

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Attrition'],axis=1)
train_y = train_data['Attrition']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Attrition'],axis=1)
test_y = test_data['Attrition']

# find out the indices of categorical variables
categorical_var = np.where(train_x.dtypes != np.float)[0]

model = CatBoostClassifier(iterations=50)

# fit the model with the training data
model.fit(train_x,train_y,cat_features = categorical_var,plot=False)

# predict the target on the train dataset
predict_train = model.predict(train_x)

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('\naccuracy_score on train dataset : {:.2f}'.format(accuracy_train*100))

# predict the target on the test dataset
predict_test = model.predict(test_x)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : {:.2f}'.format(accuracy_test*100))