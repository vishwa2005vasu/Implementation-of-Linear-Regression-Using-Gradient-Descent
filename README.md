# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Startv the program.


2.import numpy as np.


3.Give the header to the data.


4.Find the profit of population.


5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.


6.End the program.
 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1],color="cadetblue")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000) ")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  """
  take in a numpy array X,y theta and generate the cost function of using the
  in a linear regression model
  """
  m=len(y) #length of the training data
  h=x.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err) #returning ]

data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta) #call the function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history

theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1],color="cadetblue")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value)
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $" +str(round(predict2,0)))

```

## Output:
1.profit prediction





![Screenshot (41)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/ff4fe207-f99a-41ba-87f8-ec4ff2f1507d)







 2.function output




 

![Screenshot (42)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/e5e029bb-42f2-44d9-88b0-a6a46d3ff2f5)





 3.Gradient Descent



 

![Screenshot (43)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/38ac6814-ffe7-49e8-a598-cefb4bc698ce)





 4.Cost function using gradient descent


 


![Screenshot (44)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/000fb802-7a8b-4764-bd30-fc226c076cbf)





5.Linear regression using profit prediction






![Screenshot (45)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/355e9399-53dc-4330-b4b7-3b830fa1c3f6)




 6.Profit prediction for a population of 35,000






![Screenshot (78)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/1de9c869-b806-4121-9a94-f6daf241093c)






 7.Profit prediction for a population of 70,000




 
 
 
 ![Screenshot (77)](https://github.com/MaheshMuthuL/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135570619/c9780ad2-59ca-40b4-a304-a09f314aff81)









## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
