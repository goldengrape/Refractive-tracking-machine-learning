
# coding: utf-8

# # 读取WAM5500数据文件
# 

# 一组调节测试仪器WAM5500的数据文件, 例如:  
# 
# |分类|数据文件名|
# |:--|:--|
# |有病|001.csv|
# |有病|002.csv|
# |没病|003.csv|
# |有病|004.csv|
# 
# 说明文件存储在class_fname所指定的文件中. 
# 

# In[1]:


import pandas as pd
import os
import numpy as np


# In[9]:


if __name__=="__main__":
    fpath="data_sim"
    class_fname="class_sim.csv"


# ## 读取单个文件

# In[14]:


def read_raw_data(filename):
    # csv文件中, 首行数据记录了机器型号, , 测量日期, 测量起始时间, 此信息略去未读取
    # 原始数据从下一行开始
    raw_data=pd.read_csv(filename,header=None,skiprows=1,index_col=False)
    # 数据列分别表示记录时间, 记录眼别, 视标位置, 屈光, 瞳孔直径
    raw_data.columns=["time","eye","target","power","pupil"]
    # 其中眼别和视标位置看起来对学习不重要, 故略去
    raw_data=raw_data[["time","power","pupil"]]
    
    # 测量时间使用pandas datetime格式记录, 容易使用时间间隔进行数据提取, 日期不重要
    raw_data["time"]=pd.to_datetime(raw_data["time"],unit='s')
    
    # 其他数据以float方式记录, 空缺数据记录为NaN
    try:
        raw_data["power"]=pd.to_numeric(raw_data["power"].str.strip())
    except:
        raw_data["power"]=pd.to_numeric(raw_data["power"])
    try:
        raw_data["pupil"]=pd.to_numeric(raw_data["pupil"].str.strip())
    except:
        raw_data["pupil"]=pd.to_numeric(raw_data["pupil"])

    # 丢弃空缺数据所在行
    raw_data.dropna(inplace=True)
    return raw_data


# ## 按照时间裁剪
# 测量的时间通常比所需要的时间长, 因此需要按照裁剪出指定时长的数据. 
# 由于有可能存在数据点丢失, 所以指定时长的数据可能数量并不相等, 需要补齐. 

# In[15]:


def cut_by_time(df,start_time=0, duration=5):
    # 获取一段时间内的数据
    # 测量时间通常长于所需要的时间, 因此需要截取
    start_timestamp=pd.to_datetime(start_time,unit='s')
    end_timestamp=pd.to_datetime(start_time+duration,unit="s")
    df= df.where((df.time>=start_timestamp) & (df.time<=end_timestamp)).dropna()
    
    #将时间格式转换回float
    df.time=pd.to_numeric(df.time)/10e8
    return df

def padding_time(df,duration=5,redundancy=5,padding_with="last"):
    # 将数据补齐
    # 通常每秒有4-5个数据. 
    # 默认以最后一项补齐,否则以padding_with补齐
    redundancy_length=duration*5+redundancy
    add_length=redundancy_length-len(df)
    if padding_with=="last":
        padding_item=df.tail(1)
    else:
        padding_item=pd.DataFrame(np.ones((1,df.shape[1]))*padding_with,columns=df.columns)
    df2=pd.concat([padding_item]*add_length)
    new_df=df.append(df2).reset_index()[df.columns]
    return new_df



# ## 读取数据文件的整合
# 按照文件名读取数据, 然后按照时间范围截取, 清理

# In[16]:


def get_data(fpath, filename,start_time=0,duration=5,redundancy=5,padding_with="last"):
    # 按照文件名读取数据
    # 依次进行数据清理
    # 返回Numpy array
    data_filename=os.path.join(fpath,filename)
    df=read_raw_data(data_filename)
    df=cut_by_time(df,start_time=start_time, duration=duration)
    df=padding_time(df,duration=duration,redundancy=redundancy,padding_with=padding_with)
    return df.values.reshape(1,-1)


# In[17]:


# 测试
if __name__=="__main__":
    fname="WCSD0006.csv"
    data_filename=os.path.join(fpath,fname)
    raw_data=read_raw_data(data_filename)
    raw_data=cut_by_time(raw_data)
    raw_data=padding_time(raw_data)
    print(raw_data.head())
    print(raw_data.tail())


# # 读取分类文件
# 
# 分类文件是一个csv文件, 记录了每一个数据文件的分类和文件名

# In[18]:


def get_5500_data(fpath,class_fname,start_time=0,duration=5,redundancy=5,padding_with="last"):
    class_filename=os.path.join(fpath,class_fname)
    class_data=pd.read_csv(class_filename)
    X=np.vstack([get_data(fpath, data["filename"],
                          start_time=start_time,
                          duration=duration,
                          redundancy=redundancy,
                          padding_with=padding_with) 
                 for idx,data in class_data.iterrows()])
    y=class_data["class"].values#.reshape(-1,1)
    return X,y


# In[8]:


# 测试
if __name__=="__main__":
    X,y=get_5500_data(fpath,class_fname)
    print(X)
    print(y)

