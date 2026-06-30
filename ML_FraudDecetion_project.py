import numpay as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#file import 
file=pd.read_csv(r"C:\Users\Admin\Downloads\Creditcard\creditcard.csv")
print(file.head())

print(file.shape)
print(file['Class'].value_counts())

sb.countplot(x='Class', data=file)

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Transction Type")
plt.ylabel("Number of Transactions")


plt.show()

#normalize 
sc = StandardScaler()
file['Amount'] = sc.fit_transform(file[['Amount']])
file['Class'] = sc.fit_transform(file[['Class']])

#split data
x = file.drop('Class', axis=1)
y = file['Class']

x_train, x_test, y_train,  y_test =train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#model train
model= LogisticRegression()
model.fit(x_train,y_train)

#predict
y_pred=(model.predict(x_test))

#evaluation
print("accuracy:",accuracy_score(y_test, y_pred) )
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))