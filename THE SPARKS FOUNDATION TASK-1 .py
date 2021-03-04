#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION

# # Task-1

# # Prediction using supervised ML

# # BY:Isvariyashree

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df.head()


# In[3]:


df.describe()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[5]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


x=df.iloc[:, :-1].values
y=df.iloc[:,1].values


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train, x_test,y_train,y_test= train_test_split(x,y,train_size=0.80,test_size=0.20,random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression  
lr = LinearRegression()  
lr.fit(x_train, y_train) 

print("Training complete.")


# In[10]:


line= lr.coef_*x+lr.intercept_

plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[11]:


print(lr.score(x_train,y_train))
print("Train Score")


# In[12]:


print(lr.score(x_test,y_test))
print("Test Score")


# In[13]:


print("Predicted score if a student studies for 9.25 hours/day ",lr.predict([[9.25]]))


# In[ ]:




