import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# reading training and testing data
df = pd.read_csv('DATA (1).csv')
df.head()
df['21']=df['21'].map({1:1,2:-1,3:0})
df['22']=df['22'].map({1:1,2:0})
df['28']=df['28'].map({1:-1,2:1,3:0})
# df['23']=df['23'].map({1:1,2:-1,3:0})
# 23.24可或不可
# df['24']=df['24'].map({1:1,2:2,3:0})
# 28 有時有用
y=df['GRADE']
col=['1','4','5','6','7','8','9','10',
    '11','12','14','15','16','17','18','19','20',
    '21','22','23','24','25','26','27','28','29','30']
X=df[col]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain,ytrain)

knn_tr_acc = knn.score(xtrain, ytrain)
knn_te_acc = knn.score(xtest, ytest)
print('1-NN training acc:',knn_tr_acc)
print('1-NN testing acc:',knn_te_acc)

class_weight=[]
yhat = knn.predict(xtest)
knn_acc = []
logit_acc=[]
vot_acc=[]
nb_acc=[]
reg_acc=[]
knn_f1 = []
logit_f1=[]
vot_f1=[]
nb_f1=[]
reg_f1=[]

for i in range(10):
    class_weight.append(i)
    knn_acc.append(accuracy_score(ytest,yhat))
    knn_f1.append(f1_score(ytest, yhat, average='weighted') )
    # knn_prec.append(precision_score(ytest,yhat))
    # knn_recall.append(recall_score(ytest,yhat))
    # knn_f1.append(f1_score(ytest,yhat))
for weight in range(1,11):
    logit = LogisticRegression(C=1,class_weight={0:1,1:weight})
    logit.fit(xtrain,ytrain)
    yhat = logit.predict(xtest)
    logit_acc.append(accuracy_score(ytest,yhat))
    logit_f1.append(f1_score(ytest, yhat, average='weighted') )
for weight in range(1,11):
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1,max_iter=50)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[
            ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf1 = eclf1.fit(xtrain, ytrain)
    yhat = eclf1.predict(xtest)
    vot_acc.append(accuracy_score(ytest,yhat))
    vot_f1.append(f1_score(ytest, yhat, average='weighted') )
for weight in range(1,11):
    nb = GaussianNB()
    nb.fit(xtrain,ytrain)
    yhat = nb.predict(xtest)
    nb_acc.append(accuracy_score(ytest,yhat))
    nb_f1.append(f1_score(ytest, yhat, average='weighted') )
for weight in range(1,11):
    reg =GradientBoostingClassifier(n_estimators=100, learning_rate=5.0,
         max_depth=1, random_state=0)
    reg.fit(xtrain,ytrain)
    yhat = reg.predict(xtest)
    reg_acc.append(accuracy_score(ytest,yhat))
    reg_f1.append(f1_score(ytest, yhat, average='weighted') )
   
    
plt.subplot(121)
plt.plot(class_weight,knn_acc,'k-',label='knn_acc')
plt.plot(class_weight,logit_acc,'r-',label='logit_acc')
plt.plot(class_weight,vot_acc,'b-',label='vot_acc')
plt.plot(class_weight,nb_acc,'g-',label='nb_acc')
plt.plot(class_weight,reg_acc,'y-',label='reg_acc')
plt.xlabel('class weight')
plt.ylabel('accurancy')
plt.legend()
plt.subplot(122)
plt.plot(class_weight,knn_f1,'k-',label='knn_f1')
plt.plot(class_weight,logit_f1,'r-',label='logit_f1')
plt.plot(class_weight,vot_f1,'b-',label='vot_f1')
plt.plot(class_weight,nb_f1,'g-',label='nb_f1')
plt.plot(class_weight,reg_f1,'y-',label='reg_f1')
plt.xlabel('class weight')
plt.ylabel('f1_score')
plt.legend()
plt.show()