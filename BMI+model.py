
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# # IMPORTING DATA SET AND PRE-PROCESSING

# In[2]:


dataset=pd.read_csv('/Users/swapnilgharat/Desktop/520/FINAL PROJECT/Train.csv')
dataset.shape


# In[3]:


dataset.y.value_counts()


# In[4]:


dataset= dataset.drop('Row', axis =1)


# In[5]:


dataset.shape


# # SINCE THERE IS A UNBALANCE BETWEEN THE CLASSES WE USE UPSAMPLING TO MAKE THE NUMBER OF INSTANCES IN EACH CLASSES EQUAL

# 
# STEPS TO CONVERT CATEGORICAL DATA TO NUMERICAL DATA
# STEP1: SEPERATE AND DROP CATEGORICAL DATA
# 
# 
# 

# In[6]:



CAT=list(dataset.select_dtypes(include=['object']))
dpX=dataset.drop(CAT,axis =1).astype('float64')
dpX
list (CAT)


# In[7]:


dpX.shape 


# In[8]:


#CREATE DUMMY COLUMNS FOR CATEGORICAL DATA
d1=pd.get_dummies(dataset['x5'])
d2=pd.get_dummies(dataset['x13'])
d3=pd.get_dummies(dataset['x64'])
d4=pd.get_dummies(dataset['x65'])


# In[9]:


list(d3)


# In[10]:


Xnew=pd.concat([dpX,d1,d2,d3,d4],axis=1).astype('float')


# In[11]:


Xnew.shape


# In[12]:


import numpy as np
Xnew.fillna(np.mean(Xnew), inplace = True)
Xnew.shape


# In[13]:


y = Xnew['y']


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnew, y, test_size=0.2, random_state=42)


# # PROCESSING TRAIN DATA(UPSAMPLING)

# In[15]:


X_train.y.value_counts()


# In[16]:


#upsampling of the dataset
from sklearn.utils import resample
major = X_train[X_train['y']==-1]
minor = X_train[X_train['y']==1]
upsampled = resample(minor,replace=True,n_samples=1516,random_state=123)
newupsampled = pd.concat([major,upsampled])


# In[17]:


newupsampled.y.value_counts()


# In[18]:


ynew_train = newupsampled['y']
X = newupsampled.drop('y', axis =1)


# In[19]:


X.shape


# In[20]:


ynew_train.shape


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XStdtrain = scaler.fit(X).transform(X)


# In[22]:


XStdtrain.shape


# # TEST DATA

# In[23]:


X_test.shape


# In[24]:


ynew_test = X_test['y']
Xtest =X_test.drop('y', axis =1)


# In[25]:


Xtest.shape


# In[26]:


XStdtest = scaler.transform(Xtest)


# In[27]:


XStdtest.shape


# In[28]:


ynew_test.shape


# # TRIAL 1: USING SVM

# In[29]:


from sklearn import svm
from sklearn.svm import SVC
modelsvm = SVC()
modelsvm.fit(XStdtrain,ynew_train)
ypred = modelsvm.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(ynew_test, ypred)*100, "%")
print (metrics.classification_report(ynew_test, ypred))


# In[30]:


tuned_parameters = [{'kernel': ['rbf',], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000],'C': [1, 10, 100, 1000]}]
modsvm2 = GridSearchCV(modelsvm, cv = 5 ,refit = 'true',param_grid = tuned_parameters)
modsvm2.fit(XStdtrain,ynew_train)
print(modsvm2.best_params_)


# In[31]:


modsvm3 = SVC(kernel = 'rbf', C = 1000,gamma=0.001, probability = True,class_weight = 'balanced')


# In[32]:


XStdtest.shape


# In[33]:


ynew_test.shape


# In[34]:


modsvm3.fit(XStdtrain,ynew_train)
yprednew =modsvm3.predict(XStdtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ynew_test, yprednew))


# In[35]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ynew_test,yprednew)


# # ACCURACY: 78.6% BALANCED ERROR RATE: 0.204

# # TRIAL 2: USING DECISION TREE

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import itertools
import sklearn as skl_lm

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[37]:


modtree = DecisionTreeClassifier()
modtree.fit(XStdtrain,ynew_train)
ypredDT = modtree.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(ynew_test, ypredDT)*100, "%")
print (metrics.classification_report(ynew_test, ypredDT))


# In[38]:


tuneparameters = [{'criterion': ['entropy','gini'], 'max_depth': [1,2,3,4,5,6,7,8,9,10],'min_samples_split': [2,3,4,5,6,7,8,9,10]}]
modtree2 = GridSearchCV(modtree, cv = 8 ,refit = 'true',param_grid = tuneparameters)
modtree2.fit(XStdtrain,ynew_train)


# In[39]:


print(modtree2.best_params_)


# In[72]:


modelregtree_3= DecisionTreeClassifier(criterion ='gini',max_depth = 10,min_samples_split=4)
modelregtree_3.fit(XStdtrain,ynew_train)
ypredreg = modelregtree_3.predict(XStdtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ynew_test, ypredreg))


# In[73]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ynew_test,ypredreg)


# # ACCURACY:75.4% BALANCED ERROR RATE:0.28
# 

# # TRIAL 3:USING RANDOM FOREST

# In[42]:


from sklearn.ensemble import RandomForestClassifier
seedStart = 2357
modelnowRF=RandomForestClassifier(random_state=seedStart)
modelnowRF.fit(XStdtrain,ynew_train)


# In[43]:


ynew_test.shape


# In[44]:


ypredRF = modelnowRF.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(ynew_test, ypredRF)*100, "%")
print (metrics.classification_report(ynew_test, ypredRF))
from sklearn.metrics import confusion_matrix
confusion_matrix(ynew_test,ypredRF)


# In[45]:


from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
param_grid = {'n_estimators': [200,700], 'max_features': ['auto', 'sqrt', 'log2']}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 7)
CV_rfc.fit(XStdtrain,ynew_train)




# In[46]:


print(CV_rfc.best_params_)


# In[69]:


rfcT = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=700, oob_score = True)
rfcT.fit(XStdtrain,ynew_train)


# In[70]:


XStdtest.shape


# In[71]:


ypredrfct = rfcT.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(ynew_test, ypredrfct)*100, "%")
print (metrics.classification_report(ynew_test, ypredrfct))
from sklearn.metrics import confusion_matrix
confusion_matrix(ynew_test,ypredrfct)


# # Training Accuracy= 83.4 %
# 
# 

# # Balanced error rate = 0.504/2 = 0.252

# 
# # BUILDING FINAL MODEL ON RANDOM FOREST

# In[50]:


####TEST  MODEL


# In[6]:


testdataset=pd.read_csv('/Users/swapnilgharat/Desktop/520/FINAL PROJECT/test.csv')
testdataset.shape


# In[52]:


testdataset= testdataset.drop('Row', axis =1)


# In[53]:


CAT=list(testdataset.select_dtypes(include=['object']))
CATdpX=testdataset.drop(CAT,axis =1).astype('float64')
CATdpX
list (CAT)


# In[54]:


CATdpX.shape


# In[55]:


#CREATE DUMMY COLUMNS FOR CATEGORICAL DATA
d5=pd.get_dummies(testdataset['x5'])
d6=pd.get_dummies(testdataset['x13'])
d7=pd.get_dummies(testdataset['x64'])
d8=pd.get_dummies(testdataset['x65'])


# In[75]:


list(d8)


# In[57]:


Xfortest=pd.concat([CATdpX,d5,d6,d7,d8],axis=1).astype('float')


# In[58]:


import numpy as np
#Xnew.replace([np.inf, -np.inf], np.nan)
Xfortest.fillna(np.mean(Xtest), inplace = True)
Xfortest.shape


# In[59]:


l= ['x5B','x5D','x64Mk','x64Mm','x65N']
for col in l:
    Xfortest[col]=0


# In[60]:


Xfortest.shape


# In[61]:


Xfortest.head()


# In[62]:


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
XforStdtest = scaling.fit(Xfortest).transform(Xfortest)


# In[63]:


yforpredtest = modsvm3.predict(XforStdtest)


# In[64]:


yforpredtest


# In[76]:


print (yforpredtest)


# # EXPORTING YPREDATEST VALUES TO EXCEL

# In[77]:


import numpy as np
import pandas as pd
row_number=np.linspace(1,1647,1647)
yforpredtest_df = pd.DataFrame(row_number)


# In[78]:


yforpredtest_df['yforpredtest'] = yforpredtest


# In[79]:


yforpredtest_df.to_csv('/Users/swapnilgharat/Desktop/520/FINAL PROJECT/Classificationresults.csv',header=None,index=False)

