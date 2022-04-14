# import libraries
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# read the data
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1]

#split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#featuers scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
 
# fitting svm to the training data
svm_classifier=SVC(kernel='rbf',random_state=0)
svm_classifier.fit(x_train,y_train)
    
# predicting the test result
y_pred =svm_classifier.predict(x_test)

#score the accuracy
sx=svm_classifier.score(x_train,y_train)
sy=svm_classifier.score(x_test,y_test)
print(sx)
print(sy)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#visualize confusion matrix
sns.heatmap(cm,center=True)
plt.show()





