
# coding: utf-8

# In[1]:


# Author - Ritvik Khanna 
# Date - 04/05/18 
# Version - 2.3


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# import os
# print(os.listdir("../input"))
# Load dataset
df = pd.read_csv("dataset/heart.csv")
df.head(10)



# """
# age         age in years
# sex         (1 = male; 0 = female)
# cp          chest pain type
# trestbps    resting blood pressure (in mm Hg on admission to the hospital)
# chol        serum cholestoral in mg/dl
# fbs         (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restecg     resting electrocardiographic results
# thalach     maximum heart rate achieved
# exang       exercise induced angina (1 = yes; 0 = no)
# oldpeak     ST depression induced by exercise relative to rest
# slope       the slope of the peak exercise ST segment
# ca          number of major vessels (0-3) colored by flourosopy
# thal        3 = normal; 6 = fixed defect; 7 = reversable defect
# target      1 or 0
# """


# In[87]:


X = df.values[:, :13]
Y = df.values[:,13]
xyz=df.values[180:181,:13]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[88]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[92]:


y_pred = clf_gini.predict(X_test)
#y_pred
print ("\nAcuraccy score ::: ",accuracy_score(y_test,y_pred)*100)


# In[95]:


from sklearn.externals import joblib
filename = 'finalized_model.sav'
joblib.dump(clf_gini, filename)

