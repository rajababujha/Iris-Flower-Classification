from tkinter import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

master = Tk()

master.geometry('550x400')
answer_label =Label(master, text ="---")
answer_label.grid(row =10, column =0)

iris = pd.read_csv("iris2.csv")
##print(iris.head(5))

##print(iris.describe())
##print(iris['species'].value_counts())
##
##sns.pairplot(iris,hue='species')
##plt.show()

X = iris.drop(['species'],axis = 1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
##    print(X_train.shape)
##    print(y_train.shape)
##    print(X_test.shape)
##    print(y_test.shape)


scores = []
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores.append(metrics.accuracy_score(y_test, y_pred))
##    print(scores)
##    print([[a,b,c,d]])

##print(knn.predict([[5.1,3.5,1.4,0.2]]))
##print(knn.predict([[5.9,3.0,5.1,1.8]]))
##print(knn.predict([[3.0, 3.6, 1.3, 0.25]])) #Setosa


Label(master, text='Sepal_length').grid(row=0) 
Label(master, text='Sepal_width').grid(row=1)
Label(master, text='Petal_lenth').grid(row=2) 
Label(master, text='Petal_width').grid(row=3)
e1 = Entry(master) 
e2 = Entry(master)
e3 = Entry(master) 
e4 = Entry(master)

e1.grid(row=0, column=1) 
e2.grid(row=1, column=1)
e3.grid(row=2, column=1) 
e4.grid(row=3, column=1)




def algorithms():
    if (e1.get() and e2.get() and e3.get() and e4.get()!= ""):
        E1 = float(e1.get())
        E2 = float(e2.get())
        E3 = float(e3.get())
        E4 = float(e4.get())
        lst = np.array([E1,E2,E3,E4]).reshape(1,-1)

        answer = knn.predict(lst)
        answer_label.configure(text =str(answer))
        status_label.configure(text ="successfully computed")



##Label(master, text='Sepal_length').grid(row=0) 
##Label(master, text='Sepal_width').grid(row=1)
##Label(master, text='Petal_lenth').grid(row=2) 
##Label(master, text='Petal_width').grid(row=3)
##e1 = Entry(master) 
##e2 = Entry(master)
##e3 = Entry(master) 
##e4 = Entry(master)
##
##e1.grid(row=0, column=1) 
##e2.grid(row=1, column=1)
##e3.grid(row=2, column=1) 
##e4.grid(row=3, column=1)

calculate_button =Button(master, text="calculate", command= algorithms)
calculate_button.grid(row =7, column =0, columnspan =2)

status_label =Label(master, height =5, width =25, bg ="black", fg ="#00FF00", text ="---", wraplength =150)
status_label.grid(row =8, column =0, columnspan =2)
mainloop() 
