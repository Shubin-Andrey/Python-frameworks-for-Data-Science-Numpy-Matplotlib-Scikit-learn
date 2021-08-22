#!/usr/bin/env python
# coding: utf-8

# Тема “Обучение без учителя”
# Задание 1
# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов 
# данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.
# 

# In[9]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import load_boston


# In[3]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[2]:


boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

X.info()


# In[5]:


X = reduce_mem_usage(X)


# In[6]:


X.info()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[17]:


tsne = TSNE(n_components=2, learning_rate=250, random_state=42)

X_train_tsne = tsne.fit_transform(X_train_scaled)

print('До:\t{}'.format(X_train_scaled.shape))
print('После:\t{}'.format(X_train_tsne.shape))


# In[18]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])

plt.show()


# Диаграмма рассеивания отличается от той, что представлена в уроке. По моему предположению это из за того что learning_rate=250 и random_state=42

# Задание 2
# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# Вычислите средние значения price и CRIM в разных кластерах.
# 

# In[19]:


from sklearn.cluster import KMeans


# In[21]:


kmeans = KMeans(n_clusters=3, random_state=42, max_iter=100)

labels_train = kmeans.fit_predict(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)

plt.show()


# In[22]:


labels_train


# In[30]:


X_train


# In[29]:


X_train = X_train.assign(labels = labels_train)


# In[41]:


labels_test = kmeans.predict(X_test_scaled)


# In[42]:


X_test = X_test.assign(labels = labels_test)


# In[43]:


X_test


# In[64]:


y_train_df = pd.DataFrame(y_train, columns=('Price',))
y_test_df = pd.DataFrame(y_test, columns=('Price',))


# In[65]:


y_train_df = y_train_df.assign(labels = labels_train)
y_test_df = y_test_df.assign(labels = labels_test)


# In[85]:


print('Средние значения по приступности на обучающей выборке:')
print(X_train[['labels', 'CRIM']].groupby('labels').mean())


# In[86]:


print('Средние значения по приступности на тестовой выборке:')
print(X_test[['labels', 'CRIM']].groupby('labels').mean())


# In[87]:


print('Средние значения по цене на обучающей выборке:')
print(y_train_df[['labels', 'Price']].groupby('labels').mean())


# In[88]:


print('Средние значения по цене на тестовой выборке:')
print(y_test_df[['labels', 'Price']].groupby('labels').mean())


# *Задание 3
# Примените модель KMeans, построенную в предыдущем задании,
# к данным из тестового набора.
# Вычислите средние значения price и CRIM в разных кластерах на тестовых данных.
# 

# In[89]:


labels_test = kmeans.predict(X_test_scaled)
X_test = X_test.assign(labels = labels_test)
y_test_df = pd.DataFrame(y_test, columns=('Price',))
y_test_df = y_test_df.assign(labels = labels_test)
print('Средние значения по приступности на тестовой выборке:')
print(X_test[['labels', 'CRIM']].groupby('labels').mean())
print('Средние значения по цене на тестовой выборке:')
print(y_test_df[['labels', 'Price']].groupby('labels').mean())


# In[ ]:




