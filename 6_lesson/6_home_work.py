#!/usr/bin/env python
# coding: utf-8

# Тема “Обучение с учителем”
# 
# Задание 1
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
# 

# In[1]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()

boston.keys()


# In[9]:


data = boston["data"]
data.shape


# In[10]:


feature_names = boston["feature_names"]

feature_names


# In[11]:


target = boston["target"]

target[:10]


# In[12]:


X = pd.DataFrame(data, columns=feature_names)

X.head()


# In[13]:


X.info()


# In[14]:


Y = pd.DataFrame(target, columns=["price"])

Y.info()


# In[15]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)


# In[34]:


X_train


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


lr = LinearRegression()


# In[69]:


lr.fit(X_train, y_train)


# In[33]:


y_pred = lr.predict(X_test)

y_pred.shape


# In[35]:


check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)


# In[36]:


check_test["error"] = check_test["y_pred"] - check_test["y_test"]

check_test.head()


# In[37]:


from sklearn.metrics import r2_score


# In[56]:


R2 = r2_score(check_test["y_test"], check_test["y_pred"])
R2


# In[57]:


help(r2_score)


# Задание 2
# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
# Сделайте агрумент n_estimators равным 1000,
# max_depth должен быть равен 12 и random_state сделайте равным 42.
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression,
# но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0],
# чтобы получить из датафрейма одномерный массив Numpy,
# так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.
# 

# In[82]:


from sklearn.ensemble import RandomForestRegressor


# In[88]:


model = RandomForestRegressor(max_depth=12, max_features=6, n_estimators=1000, random_state=42)

model.fit(X_train.values, y_train.values[:, 0])


# In[89]:


y_pred = model.predict(X_test)


# In[90]:


check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)


# In[91]:


check_test["error"] = check_test["y_pred"] - check_test["y_test"]

check_test.head()


# In[92]:


R2_2 = r2_score(check_test["y_test"], check_test["y_pred"])
R2_2


# In[93]:


R2 # Значение ошибки при использовании линейной регрессии


# Вывод: алгоритм случайного леса работает на этих данных намного лутчше

# *Задание 3
# Вызовите документацию для класса RandomForestRegressor,
# найдите информацию об атрибуте feature_importances_.
# С помощью этого атрибута найдите сумму всех показателей важности,
# установите, какие два признака показывают наибольшую важность.
# 

# In[95]:


help(RandomForestRegressor)


# In[104]:


fi = pd.DataFrame({'feature': feature_names,
                   'importance': model.feature_importances_}).sort_values('importance', ascending = False)


# In[105]:


fi


# In[102]:


sum_fi = fi['importance'].sum()
sum_fi


# In[103]:


fi.iloc[0:2]


# *Задание 4
# В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию по библиотеке Matplotlib, это датасет Credit Card Fraud Detection.Для этого датасета мы будем решать задачу классификации - будем определять,какие из транзакциции по кредитной карте являются мошенническими.Данный датасет сильно несбалансирован (так как случаи мошенничества относительно редки),так что применение метрики accuracy не принесет пользы и не поможет выбрать лучшую модель.Мы будем вычислять AUC, то есть площадь под кривой ROC.
# Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.
# Загрузите датасет creditcard.csv и создайте датафрейм df.
# С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка несбалансирована. Используя метод info, проверьте, все ли столбцы содержат числовые данные и нет ли в них пропусков.Примените следующую настройку, чтобы можно было просматривать все столбцы датафрейма:
# pd.options.display.max_columns = 100.
# Просмотрите первые 10 строк датафрейма df.
# Создайте датафрейм X из датафрейма df, исключив столбец Class.
# Создайте объект Series под названием y из столбца Class.
# Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы: test_size=0.3, random_state=100, stratify=y.
# У вас должны получиться объекты X_train, X_test, y_train и y_test.
# Просмотрите информацию о их форме.
# Для поиска по сетке параметров задайте такие параметры:
# parameters = [{'n_estimators': [10, 15],
# 'max_features': np.arange(3, 5),
# 'max_depth': np.arange(4, 7)}]
# Создайте модель GridSearchCV со следующими аргументами:
# estimator=RandomForestClassifier(random_state=100),
# param_grid=parameters,
# scoring='roc_auc',
# cv=3.
# Обучите модель на тренировочном наборе данных (может занять несколько минут).
# Просмотрите параметры лучшей модели с помощью атрибута best_params_.
# Предскажите вероятности классов с помощью полученнной модели и метода predict_proba.
# Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и запишите в массив y_pred_proba. Из модуля sklearn.metrics импортируйте метрику roc_auc_score.
# Вычислите AUC на тестовых данных и сравните с результатом,полученным на тренировочных данных, используя в качестве аргументов массивы y_test и y_pred_proba.
# 

# In[106]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[108]:


df = pd.read_csv("D:\creditcard.csv")
df


# In[112]:


df.value_counts(normalize=True)


# In[113]:


df.info()


# In[114]:


pd.options.display.max_columns = 100


# In[115]:


df.head(10)


# In[116]:


X = df.drop('Class', axis=1)
X


# In[117]:


y = df.Class
y


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100, stratify=y)


# In[121]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[122]:


parameters = [{'n_estimators': [10, 15],
              'max_features': np.arange(3, 5),
              'max_depth': np.arange(4, 7)}]


# In[123]:


model = GridSearchCV(
        estimator=RandomForestClassifier(random_state=100),
        param_grid=parameters,
        scoring='roc_auc',
        cv=3
        )


# In[124]:


model.fit(X_train, y_train)


# In[125]:


model.best_params_


# In[133]:


y_pred_proba = model.predict_proba(X_test)
y_pred_proba


# In[134]:


y_pred_proba = y_pred_proba[:, 1]
y_pred_proba


# In[135]:


from sklearn.metrics import roc_auc_score


# In[136]:


roc_auc_score(y_test, y_pred_proba) 


# *Дополнительные задания:
# 

# 1). Загрузите датасет Wine из встроенных датасетов sklearn.datasets с помощью функции load_wine в переменную data.

# In[138]:


from sklearn.datasets import load_wine


# In[139]:


data = load_wine()


# In[140]:


data.keys()


# 2). Полученный датасет не является датафреймом. Это структура данных, имеющая ключи аналогично словарю. Просмотрите тип данных этой структуры данных и создайте список data_keys, содержащий ее ключи.
# 

# In[141]:


type(data)


# In[142]:


data_keys = data.keys()


# 3). Просмотрите данные, описание и названия признаков в датасете. Описание нужно вывести в виде привычного, аккуратно оформленного текста, без обозначений переноса строки, но с самими переносами и т.д.4). Сколько классов содержит целевая переменная датасета? Выве
# дите названия классов.
# 

# In[143]:


print(data['DESCR'])


# In[144]:


print(data['feature_names'])


# In[147]:


print(data['target_names'])
print(len(data['target_names']))


# 5). На основе данных датасета (они содержатся в двумерном массиве Numpy) и названий признаков создайте датафрейм под названием X.
# 

# In[151]:


X = pd.DataFrame(data['data'], columns=data['feature_names'])

X.head(10)


# 6). Выясните размер датафрейма X и установите, имеются ли в нем пропущенные значения.
# 

# In[152]:


X.shape


# In[154]:


X.info()


# Пропущенных значений нет

# 7). Добавьте в датафрейм поле с классами вин в виде чисел, имеющих тип данных numpy.int64. Название поля - 'target'.
# 

# In[155]:


X['target'] = data['target']


# In[158]:


X.head(10)


# In[168]:


X.info()


# In[167]:


X['target'] = X['target'].astype(np.int64)


# 8). Постройте матрицу корреляций для всех полей X. Дайте полученному датафрейму название X_corr.

# In[172]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[173]:


plt.figure(figsize = (30,20))

sns.set(font_scale=1.4)

X_corr = X.corr()
X_corr = np.round(X_corr, 2)
X_corr[np.abs(X_corr) < 0.3] = 0

sns.heatmap(X_corr, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# 9). Создайте список high_corr из признаков, корреляция которых с полем target по абсолютному значению превышает 0.5 (причем, само поле target не должно входить в этот список).
# 

# In[195]:


high_corr = X_corr[np.abs(X_corr) > 0.5]
high_corr = high_corr['target'][:-1]
high_corr.dropna(inplace=True)
high_corr


# In[199]:


high_corr = X_corr.loc[np.abs(X_corr['target']) > 0.5, 'target']
high_corr = high_corr[:-1]
high_corr


# 10). Удалите из датафрейма X поле с целевой переменной. Для всех признаков, названия которых содержатся в списке high_corr, вычислите квадрат их значений и добавьте в датафрейм X соответствующие поля с суффиксом '_2', добавленного к первоначальному названию признака. Итоговый датафрейм должен содержать все поля, которые, были в нем изначально, а также поля с признаками из списка high_corr, возведенными в квадрат. Выведите описание полей датафрейма X с помощью метода describe.

# In[203]:


X.drop('target',axis=1, inplace=True)


# In[216]:


for el in high_corr.index:
    el_str = el + '_2'
    print(el_str)
    X[el_str] = X[el] * X[el]
    print(X[el_str])


# In[217]:


X


# In[219]:


X.describe()


# In[ ]:




