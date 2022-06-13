import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing data and data preprocessing step
data = pd.read_csv('heart.csv')
print(data.head())
y = data['target'] #output

X = data.drop(['age', 'sex', 'cp','target','fbs', 'restecg','exang','slope', 'ca', 'thal'], axis=1) #input

print(X.head())



#data normalization

def normalization(X):               # Helper function to normalize data
    z = (X - np.min(X)) / (np.max(X) - np.min(X))
    return z

X = normalization(X)

print(X.head()) #after normalization
 

#*****************************************************************************************************

def hypothesis(X, theta):            # sigmoid function = hypothesis function 
    z = np.dot(theta, X.T)
    S = 1/(1+np.exp((-z)))     
    return S


def cost(X, y, theta):
    #he cost function used to minimize the error of our model
    N = len(X)
    A = hypothesis(X, theta)
    costResult = -(1/N) * np.sum((y*np.log(A)) + ((1-y)*np.log(1-A)))
    return costResult

def gradient_descent(X, y, theta, alpha, n_iter):
    m =len(X)
    c = cost(X, y, theta)
    costs = [c]
    
    for i in range(0, n_iter):
        hypo = hypothesis(X, theta)
        for i in range(X.shape[1]):
            grad_val = (1/m) * np.sum((hypo-y)*X.iloc[:, i]) # iloc >>> all rows with i column only (integer column access)
            theta[i] -= alpha * grad_val
            
        costs.append(cost(X, y, theta))
    return costs, theta


def predict(X, y, theta, alpha, n_iter):
    costs, theta = gradient_descent(X, y, theta, alpha, n_iter) 
    hypo = hypothesis(X, theta)
    Hnum = len(hypo)
    
    for i in range(Hnum):
        hypo[i]=1 if hypo[i]>=0.5 else 0
        
    y = list(y)
    yNum = len(y)
    accuracy = np.sum([y[i] == hypo[i] for i in range(yNum)])/yNum
    return costs, accuracy


#***************************************************************************************************


theta = np.zeros(X.shape[1])       # Initial values for theta with zeros for the first row with 4 columns
theta = np.random.rand(X.shape[1]) #random ones
n_iter = 2000                      # Initial values for number of iterations
alpha = 0.5                   # Initial values for learning rate

costs, accuracy = predict(X, y, theta, alpha, n_iter)       #value of prediction returned in costs and acc

#values
print("\n\nmaximum accuracyof the model is (The error of the model): ", accuracy)
print("maximum accuracy percentage of the model is (The error of the model): ", accuracy*100, "%")
print("\n\nminimum cost of the model is: ", costs[-1])          #last value in the array of costs
print("cost array of the model is: ", costs)


#plotting the graph\

#dimensions of the graph
plt.figure(figsize = (10, 7))

plt.scatter(range(0, len(costs)), costs, color ="red")      # [number of iterations = range(0, len(costs))]    Vs.   costs 

#labels
plt.xlabel("NUMBER OF ITERATIONS", color = "blue", size= 18)  #name of x-axis
plt.ylabel("COST", color = "blue", size= 18)                  #name of y-axis
plt.show()