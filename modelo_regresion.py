import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.linear_model import LogisticRegression






def regresionLineal(data=None):
    if data==None:
        data=pd.read_csv("Basedatos/salarios.csv")
    Paises=[0,1,2,3,4]
    n=len(data)
    Paises=[random.randint(0,4) for i in range(0,n)]
    data["Pais"]=Paises
    x=data[["Aexperiencia","Pais"]]
    y=data["Salario"]
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)

    fig = plt.figure(dpi = 150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train['Aexperiencia'],X_train['Pais'],Y_train,c='b',marker='s')
    ax.scatter(X_train['Aexperiencia'],X_train['Pais'],regressor.predict(X_train),c='r',marker='o')
    ax.set_xlabel('$Años de experiencia$')
    ax.set_ylabel('$Pais Normalizado$')
    ax.set_zlabel('$Salario$')
    plt.show()
    return(regressor.score(X_test,Y_test))

def regresionLogistica(data=None):
    if data==None:
        diabetes=pd.read_csv("Basedatos/diabetes.csv")
    return(diabetes)




if __name__== "__main__":
    print(regresionLogistica())
