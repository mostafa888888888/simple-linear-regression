import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

data=pd.read_csv("D:\pythonProject\predict-salary\Salary_Data.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)                                                    
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
y_pred_trai=regressor.predict(x_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,y_pred_trai,color='blue')
plt.title("salary,experience")
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,y_pred_trai,color='blue')
plt.title("salary,experience")
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
