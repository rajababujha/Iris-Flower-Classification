import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


'''downlaod iris.csv from https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'''
#Load Iris.csv into a pandas dataFrame.
iris = pd.read_csv("iris2.csv")
print(iris.head(5))

##print(iris.describe())
##print(iris['species'].value_counts())
##
##sns.pairplot(iris,hue='species')
##plt.show()

X = iris.drop(['species'],axis = 1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


scores = []
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)


print(knn.predict([[5.1,3.5,1.4,0.2]]))
print(knn.predict([[5.9,3.0,5.1,1.8]]))
print(knn.predict([[3.0, 3.6, 1.3, 0.25]])) #Setosa


