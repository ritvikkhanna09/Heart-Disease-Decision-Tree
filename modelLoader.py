
# coding: utf-8

# In[23]:



from sklearn.externals import joblib
import pandas as pd


# In[24]:


df = pd.read_csv("prediction/value_to_predict.csv")
df


# In[25]:


to_predict = df.values[:1]
to_predict


# In[27]:


loaded_model = joblib.load('finalized_model.sav')
result = loaded_model.predict(to_predict)
print("Predicted label :::: ", result)


# In[34]:


f= open("predicted.txt","w+")
f.write(str(int(result[0])))
f.close()

