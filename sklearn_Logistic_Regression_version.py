import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("heart.csv")

print(data.info()) #data and its types to under stand data

print(data.head()) #first 5 rows of the data

print (data.shape) # output the shape as number of rows and columns
#or by the following line
print("\n\nThe dataset contains {} rows and {} columns".format(data.shape[0], data.shape[1]))



"""
Split features and labels

We have 14 columns in our dataset. "target" is the column we will predict. It is called label.
It is separated from the rest of the dataset.
After this, we will have X with 13 columns called "features" & y with a single column called "label"
"""

y = data['target'] #output

X = data.drop(['age', 'sex', 'cp','target','fbs', 'restecg','exang','slope', 'ca', 'thal'], axis=1) #input

print(X.head()) #print first 5 rows after splitting


"""
Scaling the numbers between 0 and 1. As the algorithm better understands scaled values to produce better accuracy
"""

#Scaling and normalization

features = ['trestbps', 'chol', 'thalach', 'oldpeak']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(X)
# transform data
X_scaled[features] = scaler.fit_transform(X_scaled[features])
print("\n\n\n\nThe values after scaling\n\n\n",X_scaled.head())

#Split training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, random_state=10)


#Model fitting and prediction

#fitting
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)


#Calculate accuracy score
percent = accuracy_score(y_pred, y_test)

print("\n\n\nAccuracy of the model in decimal values is (The error of the model):", percent)
print("\n\n\nAccuracy percentage of the model is (The error of the model):", percent*100, "%")