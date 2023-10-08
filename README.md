# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.
2. Type the required program.
3. Print the program.
4. End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vishwa vasu. R
RegisterNumber:212222040183
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![ex 3 1](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/04262d12-3d0e-4cc4-bf55-a63769c4c2c6)
![ex 3 2](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/1d2c5b3c-6fa8-41ce-9a61-c0143382463a)
![ex 3 3](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/d25efe95-f0c7-4207-89b7-8b2fc7efc0b8)
![ex 3 4](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/66f43599-e618-4723-9025-04d5ed6accce)
![ex 3 5](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/5a0920d1-4170-43ea-bf6a-0d24ed0944eb)
![ex 3 6](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/6c63fce9-a2e1-4d4f-b9f6-14f3a938f7f0)
![ex 3 7](https://github.com/vishwa2005vasu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135954202/7a63489d-be03-48f8-9193-2356f643059c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
