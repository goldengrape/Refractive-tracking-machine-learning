
# coding: utf-8

# # 产生模拟WAM5500数据文件

# In[1]:


N=200
class_N=2
fpath="data_sim"
class_filename="class_sim.csv"


# In[2]:


import numpy as np
import pandas as pd
import os


# ## 产生分类文件

# In[3]:


def generate_class_file(fpath,class_filename,N,class_N):
    df_class=pd.DataFrame()
    df_class["class"]=np.random.randint(class_N, size=N)
    df_class["filename"]=["WCSD"+'{:04}'.format(i)+".csv" for i in range(N)]
    df_class.to_csv(os.path.join(fpath,class_filename),index=False)
    return


# In[4]:


generate_class_file(fpath,class_filename,N,class_N)


# # 产生单个记录文件

# ## 产生time数据

# In[5]:


def build_time_data(n_data):
    base_data=np.random.randn(n_data,1)*0.05+0.2
    time_data=np.cumsum(base_data)
    time_data[0]=0
    return time_data


# ## 产生power数据

# In[6]:


def build_power_data(n_data, mean=-4.5, std=0.1):
    base_data=np.random.randn(n_data,1)*std+mean
    power_data=base_data
    return power_data


# ## 产生pupil数据

# In[7]:


def build_pupil_data(n_data,mean=3.5, std=0.1):
    base_data=np.random.randn(n_data,1)*std+mean
    pupil_data=base_data
    return pupil_data


# ## 产生首行数据

# In[8]:


def build_head_row_data():
    return ["WAM5500","DATA-000","2018/5/3","16:30:05","    "]


# ## 产生单个csv文件所需数据库

# In[9]:


def build_data_df(this_class):
    n_data=np.random.randint(30,40)
    
    if this_class == 0:
        power_mean=-4
        power_std=0.2
        pupil_mean=6
        pupil_std=0.1
    else:
        power_mean=-4.5
        power_std=0.3
        pupil_mean=3.5
        pupil_std=0.1
    time_data=build_time_data(n_data)
    power_data=build_power_data(n_data, mean=-4.5, std=0.1)
    pupil_data=build_pupil_data(n_data,mean=3.5, std=0.1)
    df=pd.DataFrame()
    df["WAM5500"]=time_data
    df["DATA-000"]="L"
    df["2018/5/3"]="OFF"
    df["16:30:05"]=power_data
    df[" "]=pupil_data
    return df


# # 根据分类文件存储数据文件

# In[10]:


df_class = pd.read_csv(os.path.join(fpath,class_filename))
for idx,class_data in df_class.iterrows():
    this_class=class_data["class"]
    this_filename=class_data["filename"]
    df_data=build_data_df(this_class)
    df_data.to_csv(os.path.join(fpath,this_filename),index=False)

