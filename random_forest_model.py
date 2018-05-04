
# coding: utf-8

# # 随机森林分类

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from read_5500_data import get_5500_data


# ## 读取数据

# In[2]:


data_path="data_sim"
class_file="class_sim.csv"
X,y=get_5500_data(data_path,class_file)


# ## 分拆数据

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## 建立并训练模型

# In[4]:


y_train.shape


# In[5]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# ## 评估模型

# In[6]:


clf.score(X_test,y_test)


# In[7]:


y_pred=clf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred)
print("AUC=",auc(fpr_rf, tpr_rf))


# ## ROC曲线

# In[8]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

