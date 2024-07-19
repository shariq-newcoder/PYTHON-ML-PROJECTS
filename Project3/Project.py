import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

print("Dataset:")
dataset = pd.read_csv('lung cancer.csv')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
pyplot.show()

A = dataset[dataset.Result == 1]
B = dataset[dataset.Result == 0]

plt.scatter(A.Age, A.Smokes, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Smokes, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Smokes")
plt.legend()
plt.title("Smokes vs Age")
plt.show()

plt.scatter(A.Age, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alcohol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Age")
plt.show()

plt.scatter(A.Smokes, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Smokes, B.Alcohol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Smokes")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Smokes vs Alcohol")
plt.show()

# split dataset
x = dataset.iloc[:, 3:5]
y = dataset.iloc[:,6]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)


# Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


print('-----------------****Using KNN Algorithm****----------------')
import math
a = math.sqrt(len(y_train))
print(a)


#defining a model - KNN
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')


#fit model
classifier.fit(x_train, y_train)


#Predict the test set result
y_pred = classifier.predict(x_test)
print(y_pred)


#Evaluate model
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("In Confusion Matrix:-----")
print("Position 1.1 shows the patients that don't have Cancer.")
print("Position 1.2 shows the number of patients that have higher risk of Cancer.")
print("Position 2.1 shows the Incorrect Value.")
print("Position 2.2 shows the correct number of patients that have Cancer.")

print('F1 Score : ', (f1_score(y_test, y_pred))*100)

print('ACCURACY : ', (accuracy_score(y_test, y_pred))*100)


#USing Decision tree
print("-----------------****Using Decision Tree Algorithm****----------------")
c = tree.DecisionTreeClassifier()
c.fit(x_train, y_train)
accu_train = np.sum(c.predict(x_train)==y_train) / float(y_train.size)
accu_test = np.sum(c.predict(x_test)==y_test) / float(y_test.size)
print('Classification accuracy  on train', accu_train * 100)
print('Classification accuracy  on test', accu_test*100)

