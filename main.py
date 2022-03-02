import pandas as pd
import numpy as np

# test = pd.read_csv('/Dataset/test.csv')
# test_OG = test.copy()
train = pd.read_csv('Dataset/train.csv')
train_OG = train.copy()



#Missing values filling for train
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace= True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace= True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(train,test_size=0.25)


submission = test[['Loan_ID','Loan_Status']]
submission = pd.DataFrame(submission)
submission = submission.set_index('Loan_ID')
submission.to_csv('Dataset/submission.csv')

test = pd.DataFrame(test)
test = test.set_index('Loan_ID')
test.to_csv('Dataset/test.csv')




#outlier treatment
train['LoanAmount_Log'] = np.log(train['LoanAmount'])
# train['LoanAmount_Log'].hist()
test['loanAmount_Log'] = np.log(test['LoanAmount'])

#final changes for model ready dataset

train = train.drop('Loan_ID',axis = 1)
test = test.drop('Loan_ID',axis = 1)
test_OG = test.copy()

X = train.drop('Loan_Status',axis=1)
y = train.Loan_Status

X = pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


#model using logistic regression

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train,y_train)#model trained

pred_test = model.predict(x_test)

acc = accuracy_score(y_test,pred_test)

pred_test = model.predict(test)

submission = pd.read_csv('Dataset/submission.csv')
print(submission.head())

submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_OG['Loan_ID']
submission['Loan_Status'].replace(0,'N',inplace = True)
submission['Loan_Status'].replace(1,'Y',inplace = True)
pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('goal.csv')
