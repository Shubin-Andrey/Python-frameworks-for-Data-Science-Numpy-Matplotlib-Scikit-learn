#!/usr/bin/env python
# coding: utf-8

# Курсовой проект для курса "Python для Data Science"
# 
# Выполнил Шубин Андрей Сергеевич

# Задание:
# Используя данные из обучающего датасета (train.csv), построить модель для предсказания цен на недвижимость (квартиры).
# С помощью полученной модели, предсказать цены для квартир из тестового датасета (test.csv).
# 
# Целевая переменная:
# Price
# 
# Метрика качества:
# R2 - коэффициент детерминации (sklearn.metrics.r2_score)
# 
# Требования к решению:
# 1. R2 > 0.6
# 2. Тетрадка Jupyter Notebook с кодом Вашего решения, названная по образцу {ФИО}_solution.ipynb, пример SShirkin_solution.ipynb
# 3. Файл CSV с прогнозами целевой переменной для тестового датасета, названный по образцу {ФИО}_predictions.csv, пример SShirkin_predictions.csv 
# Файл должен содержать два поля: Id, Price и в файле должна быть 5001 строка (шапка + 5000 предсказаний).
# 

# Загружаем необходимые модули:

# In[3612]:


import numpy as np
import pandas as pd
import pickle
import random

from scipy.spatial.distance import cdist
from scipy import stats
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.preprocessing import StandardScaler, RobustScaler

# Кластеризация
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.neighbors import KNeighborsClassifier

# Понижения размерности
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Загружаем заранее подготовленные функции

# In[3613]:


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


# In[3614]:


def emissions_del(signs):
    """Удаление выбросов"""
    max_value = train_df[signs].agg('std') * 3 + train_df[signs].agg('mean')
    print(max_value)
    if train_df[signs].agg('median') == 0:
        my_mm = round(train_df[signs].agg('mean'))
        train_df.loc[train_df[signs] > max_value, [signs]] = my_mm
        test_df.loc[test_df[signs] > max_value, [signs]] = my_mm
    else:
        my_mm = round(train_df[signs].agg('median'))
        train_df.loc[train_df[signs] > max_value, [signs]] = my_mm
        test_df.loc[test_df[signs] > max_value, [signs]] = my_mm
    train_df[signs].fillna(my_mm, inplace = True)
    test_df[signs].fillna(my_mm, inplace = True)


# In[3615]:


def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
    """
    # дописать документация
    """
    print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
    print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))

    plt.figure(figsize=(18,10))
    plt.subplot(121)
    sns.scatterplot(x=train_pred_values, y=train_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Train sample prediction')
    
    plt.subplot(122)
    sns.scatterplot(x=test_pred_values, y=test_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Test sample prediction')

    plt.show()


# Загружаем данные
# 

# In[3616]:


train_df = pd.read_csv("D:/train.csv")
train_df.head(5)


# In[3617]:


test_df = pd.read_csv("D:/test.csv")
test_df.head(5)


# In[3618]:


train_df = reduce_mem_usage(train_df)


# In[3619]:


test_df = reduce_mem_usage(test_df)


# EDA

# 1 Исследуем проблему, а именно как образуется целевая переменная и что на нее сильнн всего влияет по моему мнению
# 

# Описание датасета:
# 
# Id - идентификационный номер квартиры
# 
# DistrictId - идентификационный номер района
# 
# Rooms - количество комнат
# 
# Square - площадь
# 
# LifeSquare - жилая площадь
# 
# KitchenSquare - площадь кухни
# 
# Floor - этаж
# 
# HouseFloor - количество этажей в доме
# 
# HouseYear - год постройки дома
# 
# Ecology_1, Ecology_2, Ecology_3 - экологические показатели местности
# 
# Social_1, Social_2, Social_3 - социальные показатели местности
# 
# Healthcare_1, Helthcare_2 - показатели местности, связанные с охраной здоровья
# 
# Shops_1, Shops_2 - показатели, связанные с наличием магазинов, торговых центров
# 
# Price - цена квартиры

# Из колличественных переменных по моему мнению наиболее важные : LifeSquare, HouseYear. А из категориальных признаков переменных DistrictId,
# Shops, DistrictId, Ecology.    

# Теперь необходимо происследовать целевую переменную

# In[3620]:


train_df.info()


# In[3621]:


test_df.info()


# колличественные признаки

# In[3622]:


train_df.hist(figsize=(25,25), bins=20, grid=False);


# Категориальные признаки

# In[3623]:


cat_colnames = train_df.select_dtypes(include='category').columns.tolist()
cat_colnames


# In[3624]:


for cat_colname in cat_colnames[2:]:
    print(str(cat_colname) + '\n\n' + str(train_df[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')


# In[3625]:


target_mean = round(train_df.Price.mean(), 2)
target_median = train_df.Price.median()
target_mode = train_df.Price.mode()[0]


# In[3626]:


plt.figure(figsize = (16, 8))

sns.distplot(train_df.Price, bins=50)

y = np.linspace(0, 0.000005, 10)
plt.plot([target_mean] * 10, y, label='mean', linestyle=':', linewidth=4)
plt.plot([target_median] * 10, y, label='median', linestyle='--', linewidth=4)
plt.plot([target_mode] * 10, y, label='mode', linestyle='-.', linewidth=4)

plt.title('Price')
plt.legend()
plt.show()


# Очень странно ведет себя мода, надо разбиратся

# In[3627]:


train_df.Price.mode()[0]


# In[3628]:


train_df.Price.value_counts()


# Всех значений по 1й штуке, следоватьльно мода не информативна для работы с целевой переменной

# In[3629]:


train_df.Price.describe()


# Максимальное значения сильно выходят за рамки, необходимо проверить

# In[3630]:


train_df[train_df.Price > 500000].count()


# In[3631]:


print("Skewness: %f" % train_df['Price'].skew())
print("Kurtosis: %f" % train_df['Price'].kurt())


# Данные значения скорее всего относятся к престижным районам. Следоватьльно придется разбивать задачу на кластеры для повышения качества модели

# In[3632]:


plt.figure(figsize = (30,20))

sns.set(font_scale=1.4)

corr_matrix = train_df.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# Начинаем понемногу исследовать признаки и устранять явные ошибки в данных

# 0 DistrictId 

# In[3633]:


vc = train_df['DistrictId'].value_counts()
vc.tail(15)


# In[3634]:


repeated = set(vc[vc < 10].index.values)


# In[3635]:


train_df.loc[train_df['DistrictId'].isin(repeated), 'DistrictId'] = 510


# In[3636]:


repeated = train_df['DistrictId'].values
repeated = set(repited)


# In[3637]:


test_df.loc[~test_df['DistrictId'].isin(repeated), 'DistrictId'] = 510


# In[3638]:


test_df['DistrictId'].isin(repeated)


# In[3639]:


plt.figure(figsize = (16, 8))

sns.distplot(train_df.DistrictId, bins=50)

y = np.linspace(0, 0.000005, 10)
plt.title('DistrictId')
plt.legend()
plt.show()


# Здесь до нормального распределения далеко, данный признак является категориальным

# In[3640]:


sns.scatterplot(train_df.DistrictId, train_df.Price)


# In[3641]:


train_df['DistrictId']  = train_df['DistrictId'].astype('category')
test_df['DistrictId']  = test_df['DistrictId'].astype('category')


# In[3642]:


train_df['DistrictId'].value_counts()


# In[3643]:


test_df.loc[~test_df['DistrictId'].isin(repited), 'DistrictId']


# In[3644]:


test_df['DistrictId'].value_counts()


# In[3645]:


med_price_by_district = train_df.groupby(['DistrictId'], as_index=False).agg({'Price':'median'}).rename(columns={'Price':'MedPriceByDistrict'})

#med_price_by_district = med_price_by_district['Price'].median()

med_price_by_district.head(10)


# In[3646]:


test_df = test_df.merge(med_price_by_district, on=['DistrictId'], how='left')
train_df = train_df.merge(med_price_by_district, on=['DistrictId'], how='left')
train_df.head()


# In[3755]:


signs = 'MedPriceByDistrict'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# матрица корриляции показала, что значимыми колличественными признаками являются колличество комнат и площадь

# удаляем явные выбросы.

# In[3647]:


my_1_1 = train_df['Square'].quantile(.995)
my_1_2 = train_df['Square'].quantile(.01)

my_2_1 = train_df['LifeSquare'].quantile(.99)
my_2_2 = train_df['LifeSquare'].quantile(.01)

my_3_1 = train_df['KitchenSquare'].quantile(.99)
my_3_2 = train_df['KitchenSquare'].quantile(.01)


# In[3648]:


my_1_1


# In[3649]:


my_median1 = train_df['Square'].median()
my_median2 = train_df['LifeSquare'].median()
my_median3 = train_df['KitchenSquare'].median()


# In[3650]:


train_df.loc[(train_df['Square'] > my_1_1) | (train_df['Square'] < my_1_2), 'Square'] = my_median1

train_df.loc[(train_df['LifeSquare'] > my_2_1) | (train_df['Square'] < my_2_2), 'LifeSquare'] = my_median2

train_df.loc[(train_df['KitchenSquare'] > my_3_1) | (train_df['Square'] < my_3_2), 'KitchenSquare'] = my_median3
         


# In[3651]:


test_df.loc[(test_df['Square'] > my_1_1) | (test_df['Square'] < my_1_2), 'Square'] = my_median1

test_df.loc[(test_df['LifeSquare'] > my_2_1) | (test_df['Square'] < my_2_2), 'LifeSquare'] = my_median2

test_df.loc[(test_df['KitchenSquare'] > my_3_1) | (test_df['Square'] < my_3_2), 'KitchenSquare'] = my_median3
         


# In[3652]:


signs = 'Square'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# Комнаты. вообще странно, что число комнат представлено дробным числом

# In[3653]:


signs = 'Rooms'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[3654]:


train_df['Rooms'].value_counts()


# In[3655]:


signs = 'Rooms'
grid = sns.jointplot(train_df['Square'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[3656]:


train_df.loc[train_df['Rooms'] > 9 ]


# In[3657]:


test_df['Rooms'].value_counts()


# In[3658]:


test_df.loc[test_df['Rooms'] > 9 ]


# 19 комнат на такой мальенькой площади вряд ли подойдут даже для собак

# In[3659]:


my_median = int(train_df['Rooms'].median())
train_df.loc[train_df['Rooms'].isin([0, 10, 19]), 'Rooms'] = my_median
test_df.loc[test_df['Rooms'].isin([0, 17]), 'Rooms'] = my_median


# In[3660]:


train_df['Rooms']  = train_df['Rooms'].astype('int8')
test_df['Rooms']  = test_df['Rooms'].astype('int8')


# 3   LifeSquare 

# In[3661]:


signs = 'LifeSquare'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[3662]:


train_df['LifeSquare'].describe()


# In[3663]:


mask = train_df["Square"] < train_df["KitchenSquare"] + train_df["LifeSquare"]
train_df.loc[mask].describe()


# убираем выбросы по нижней границе

# In[3664]:


train_df.loc[train_df['LifeSquare'] < 10, 'LifeSquare'] = 10
train_df.loc[train_df['KitchenSquare'] < 3, 'KitchenSquare'] = 3


# In[3665]:


test_df.loc[test_df['LifeSquare'] < 10, 'LifeSquare'].sum()


# In[3666]:


test_df.loc[test_df['KitchenSquare'] < 3, 'KitchenSquare'].sum()


# In[3667]:


test_df.loc[test_df['LifeSquare'] < 10, 'LifeSquare'] = 10
test_df.loc[test_df['KitchenSquare'] < 3, 'KitchenSquare'] = 3


# In[3668]:


square_med_diff = (train_df.loc[train_df['LifeSquare'].notnull(), 'Square'] -                   train_df.loc[train_df['LifeSquare'].notnull(), 'LifeSquare'] -                   train_df.loc[train_df['LifeSquare'].notnull(), 'KitchenSquare']).median()

square_med_diff


# In[3669]:


#mask = train_df["Square"] < train_df["KitchenSquare"] + train_df["LifeSquare"]
train_df.loc[mask, 'Square'] = train_df["KitchenSquare"] + train_df["LifeSquare"] + square_med_diff
train_df.loc[mask]


# есть зависимость стоимости от жилой площади, но все портят выбросы. Так же значения представленные NaN и малой жилой площадью скорее всего обозначают магазины итд. Их надо исследовать отдельно.

# In[3670]:


my_df = train_df[train_df['LifeSquare'].isna()]
my_df


# In[3671]:


my_df.describe()


# In[3672]:


test_df[test_df['LifeSquare'].isna()]


# In[3673]:


test_1cat = test_df[test_df['LifeSquare'].isna()]
train_1cat = test_df[test_df['LifeSquare'].isna()]
train_1cat


# In[3674]:


test_cat = test_df[~test_df['LifeSquare'].isna()]
train_cat = test_df[~test_df['LifeSquare'].isna()]
train_1cat


# In[3675]:


train_df['Claster1'] = 0
test_df['Claster1'] = 0


# In[3676]:


train_df.loc[train_df['LifeSquare'].isnull(), 'Claster1'] = 1
test_df.loc[test_df['LifeSquare'].isnull(), 'Claster1'] = 1


# In[3677]:


train_df.loc[train_df['LifeSquare'].isnull(), 'LifeSquare'] =train_df.loc[train_df['LifeSquare'].isnull(), 'Square'] -train_df.loc[train_df['LifeSquare'].isnull(), 'KitchenSquare'] -square_med_diff


# In[3678]:


test_df.loc[test_df['LifeSquare'].isnull(), 'LifeSquare'] =test_df.loc[test_df['LifeSquare'].isnull(), 'Square'] -test_df.loc[test_df['LifeSquare'].isnull(), 'KitchenSquare'] -square_med_diff


# Исследуем второй признак с пропусками

# 14  Healthcare_1

# In[3679]:


signs = 'Healthcare_1'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[3680]:


train_df['Healthcare_1'].describe()


# In[3681]:


my_median = int(train_df['Healthcare_1'].median())


# In[3682]:


train_df['Healthcare_1'].fillna(my_median, inplace=True)
test_df['Healthcare_1'].fillna(my_median, inplace=True)


# In[ ]:





# KitchenSquare - площадь кухни

# In[3683]:


signs = 'KitchenSquare'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# Floor и HouseFloor

# In[3684]:


train_df['HouseFloor'].sort_values().unique()


# In[3685]:


train_df['Floor'].sort_values().unique()


# In[3686]:


my_median = int(train_df['HouseFloor'].median())
my_median


# In[3687]:


train_df.loc[train_df['HouseFloor'] == 0, 'HouseFloor'] = my_median


# In[3688]:


test_df.loc[test_df['HouseFloor'] == 0, 'HouseFloor'] = my_median


# In[3689]:


signs = 'Floor'
grid = sns.jointplot(train_df['Price'], train_df.loc[train_df[signs] > 0, signs], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[3690]:


np.random.seed(27)
floor_outliers = train_df[train_df['Floor'] > train_df['HouseFloor']].index
print(len(floor_outliers))

train_df.loc[floor_outliers, 'Floor'] = train_df.loc[floor_outliers, 'HouseFloor'].apply(
    lambda x: random.randint(1, x)
)


# In[ ]:





# год постройки

# In[3691]:


train_df['HouseYear'] = train_df['HouseYear'] // 10 * 10
test_df['HouseYear'] = test_df['HouseYear'] // 10 * 10


# In[3692]:


var = 'HouseYear'
data = pd.concat([train_df['Price'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="Price", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# тут мы можем наблюдать выбросы, так как я сомневаюсь в том что 4960год уже прошел. К тому же мне кажется что это категориальный признак

# In[3693]:


train_df.loc[train_df['HouseYear'] > 2020, 'HouseYear'] = 2020 #train_df['HouseYear'].median()


# In[3694]:


test_df.loc[test_df['HouseYear'] > 2020, 'HouseYear'].count()


# In[3695]:


test_df.loc[test_df['HouseYear'] < 1910, 'HouseYear'].count()


# In[3696]:


test_df.loc[test_df['HouseYear'] < 1910, 'HouseYear'] = 1910


# In[3697]:


sns.histplot(test_df.HouseYear, bins=50)


# In[3698]:


train_df['HouseYear']  = train_df['HouseYear'].astype('category')
test_df['HouseYear']  = test_df['HouseYear'].astype('category')


# In[3699]:


train_df['HouseYear'].value_counts()


# In[3700]:


test_df['HouseYear'].value_counts()


# Переработаем категориальные признаки

# In[3701]:


train_df['Ecology_2_bin'] = train_df['Ecology_2'].replace({'A':0, 'B':1})
train_df['Ecology_3_bin'] = train_df['Ecology_3'].replace({'A':0, 'B':1})
train_df['Shops_2_bin'] = train_df['Shops_2'].replace({'A':0, 'B':1})


# In[3702]:


test_df['Ecology_2_bin'] = test_df['Ecology_2'].replace({'A':0, 'B':1})
test_df['Ecology_3_bin'] = test_df['Ecology_3'].replace({'A':0, 'B':1})
test_df['Shops_2_bin'] = test_df['Shops_2'].replace({'A':0, 'B':1})


# Экологические категориальные признаки сильно коррелируют друг с другом

# In[3703]:


train_df.Ecology_2 = train_df.Ecology_2.astype('str')
train_df.Ecology_3 = train_df.Ecology_3.astype('str')
test_df.Ecology_2 = test_df.Ecology_2.astype('str')
test_df.Ecology_3 = test_df.Ecology_3.astype('str')


# In[3704]:


train_df["Eco2_Eco3"] = train_df.Ecology_2 + train_df.Ecology_3
test_df["Eco2_Eco3"] = test_df.Ecology_2 + test_df.Ecology_3
train_df["Eco2_Eco3"]


# In[3705]:


train_df.Ecology_2 = train_df.Ecology_2.astype('category')
train_df.Ecology_3 = train_df.Ecology_3.astype('category')
test_df.Ecology_2 = test_df.Ecology_2.astype('category')
test_df.Ecology_3 = test_df.Ecology_3.astype('category')

train_df.Eco2_Eco3 = train_df.Eco2_Eco3.astype('category')
test_df.Eco2_Eco3 = test_df.Eco2_Eco3.astype('category')


# In[3706]:


train_df.columns.tolist()


# In[ ]:





# In[3707]:


train_df.drop(['DistrictId', 'Ecology_2', 'Ecology_3'], axis=1, inplace=True)
test_df.drop(['DistrictId', 'Ecology_2', 'Ecology_3'], axis=1, inplace=True)


# Исследование модели на предмет кластеризации

# In[3708]:


train_df_dm = pd.get_dummies(train_df)
test_df_dm = pd.get_dummies(test_df)


# In[3710]:


train_df_dm.drop('Id', axis=1, inplace=True)
test_df_dm.drop('Id', axis=1, inplace=True)


# In[3711]:


train_df_dm


# In[3712]:


X = train_df_dm.drop('Price', axis=1)
y = train_df_dm['Price'].values

X.head(2)


# In[3713]:


train_df_dm.shape


# In[3714]:


test_df_dm.shape


# In[3715]:


X.shape


# In[3716]:


train_df_dm.columns


# In[3717]:


test_df_dm.columns


# In[3718]:


scaler = RobustScaler()

colnames = X.columns
train_df_dm_scaled = pd.DataFrame(scaler.fit_transform(X), columns=colnames)
test_df_dm_scaled = pd.DataFrame(scaler.transform(test_df_dm), columns=colnames)

train_df_dm_scaled.head(2)


# In[3719]:


#dim_reducer = TSNE(n_components=2, learning_rate=250, random_state=42, perplexity=30)
#train_df_dm_scaled = dim_reducer.fit_transform(train_df_dm_scaled.dropna())       


# In[3720]:


#plt.scatter(train_df_dm_scaled[:, 0], train_df_dm_scaled[:, 1])


# колличество кластеров по графику t-SNE = 3

# Кластеризация

# In[3721]:


kmeans_3 = KMeans(n_clusters=3, random_state=42)
labels_clast_3 = kmeans_3.fit_predict(train_df_dm_scaled)
labels_clast_3 = pd.Series(labels_clast_3, name='clusters_3')
labels_clast_3_test = kmeans_3.predict(test_df_dm_scaled)
labels_clast_3_test = pd.Series(labels_clast_3_test, name='clusters_3')

unique, counts = np.unique(labels_clast_3, return_counts=True)


# In[3722]:


clusters_3_dummies = pd.get_dummies(labels_clast_3, drop_first=True, prefix='clusters_3')

X_train_cluster = pd.concat([train_df_dm_scaled, clusters_3_dummies], 
                   axis=1)
X_train_cluster.head()


# In[3723]:


X_train_cluster.columns


# In[3724]:


clusters_3_dummies_test = pd.get_dummies(labels_clast_3_test, drop_first=True, prefix='clusters_3')

X_test_cluster = pd.concat([test_df_dm_scaled, clusters_3_dummies_test], 
                   axis=1)
X_test_cluster.head()


# In[3725]:


X_train, X_test, y_train, y_test = train_test_split(X_train_cluster, y, test_size=0.33, shuffle=True, random_state=21
)


# In[3726]:


rf_model = RandomForestRegressor(max_depth=8, min_samples_split=100, n_estimators=500, n_jobs=-1, random_state=39)
rf_model.fit(X_train, y_train)


# In[3727]:


y_train_preds = rf_model.predict(X_train)
y_test_preds = rf_model.predict(X_test)

evaluate_preds(y_train, y_train_preds, y_test, y_test_preds)


# Перекрестная проверка

# In[3728]:


cv_score = cross_val_score(
    rf_model,
    X.fillna(-9999),
    y, scoring='r2',
    cv=KFold(n_splits=5, shuffle=True, random_state=21)
)
cv_score


# In[3729]:


cv_score.mean(), cv_score.std()


# In[3730]:


cv_score.mean() - cv_score.std(), cv_score.mean() + cv_score.std()


# Важность признаков

# In[3731]:


feature_importances = pd.DataFrame(zip(X_train.columns, rf_model.feature_importances_), 
                                   columns=['feature_name', 'importance'])

feature_importances.sort_values(by='importance', ascending=False)


# In[3732]:


rf_model.fit(X_train_cluster, y)


# In[3733]:


y_test_preds = rf_model.predict(X_test_cluster)


# In[3734]:


X_test_cluster


# In[3735]:


y_test_preds.shape


# In[ ]:


y_2 =y y_test_preds


# In[3741]:


y_2 = np.concatenate( [ y, y_test_preds] , axis = 0)
y_2.shape


# In[3742]:


X_train_2 = np.concatenate( [ X_train_cluster, X_test_cluster] , axis = 0)
X_train_2.shape


# In[3744]:


y_test_preds2 = rf_model.fit(X_train_2, y_2)

y_train_preds2 = rf_model.predict(X_train_2)


# In[3745]:


print("Train R2:\t" + str(round(r2(y_2, y_train_preds2), 3)))

plt.figure(figsize=(18,10))
plt.subplot(121)
sns.scatterplot(x=y_train_preds2, y=y_2)
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.title('Train sample prediction')
    
plt.show()


# In[3739]:


test_id = test_df["Id"]
pred_df = pd.DataFrame()
pred_df["Id"] = test_id
pred_df["Price"] = y_test_preds


# In[ ]:





# In[3737]:


pred_df


# In[3766]:


assert pred_df2.shape[0] == 5000, f"Real pred-shape = {pred_df.shape[0]}, Expected pred-shape = 5000"


# In[3758]:


plt.figure(figsize = (16, 8))

sns.distplot(pred_df.Price, bins=50)

y = np.linspace(0, 0.000005, 10)

plt.title('Price')
plt.legend()
plt.show()


# In[3753]:


test_id = test_df["Id"]
pred_df2 = pd.DataFrame()
pred_df2["Id"] = test_id
pred_df2["Price"] = y_train_preds2[10000:]


# In[3762]:


pred_df2


# In[3756]:


print("Train R2:\t" + str(round(r2(y_2[:10000], y_train_preds2[:10000]), 3)))

plt.figure(figsize=(18,10))
plt.subplot(121)
sns.scatterplot(x=y_train_preds2[:10000], y=y_2[:10000])
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.title('Train sample prediction')
    
plt.show()


# In[3757]:


plt.figure(figsize = (16, 8))

sns.distplot(pred_df2.Price, bins=50)

y = np.linspace(0, 0.000005, 10)

plt.title('Price')
plt.legend()
plt.show()


# In[3760]:


r2(pred_df2.Price, pred_df.Price)


# In[3787]:


pred_df2.to_csv("D:/ASShubin_predictions", index=False)


# In[3788]:


pred_df = pd.read_csv("D:/ASShubin_predictions")
pred_df.head(n=2)


# In[ ]:




