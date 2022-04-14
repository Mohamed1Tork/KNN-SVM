#import libraries
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


# read the data 
dataset=pd.read_csv("iris.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#split data into ttrain and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# fitting knn to trainig data
knn=KNeighborsClassifier(n_neighbors=5,weights="uniform",algorithm="auto")
knn.fit(x_train,y_train)

# predicting the test result
y_pred=knn.predict(x_test)

#score the accuracy
train_acc=knn.score(x_train,y_train)
test_acc=knn.score(x_test,y_test)
print("the train accuracy",train_acc)
print("the test accuracy",test_acc)

# making confusion matrix
cm=confusion_matrix(y_test, y_pred) 


#visualize confusion matrix
sns.heatmap(cm,center=True)
plt.show
