#!/usr/bin/env python
# coding: utf-8

# Задание 1
# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно содержатся данные:  
# [1, 1, 1, 2, 2, 3, 3],
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
# [450, 300, 350, 500, 450, 370, 290].
# 

# In[1]:


import pandas as pd


# Через словарь

# In[2]:


authors_dict = {'author_id': [1, 2, 3], 'author_name' :  ['Тургенев', 'Чехов', 'Островский']}


# In[9]:


authors = pd.DataFrame(authors_dict)
authors


# Явное создание

# In[8]:


book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3], 
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                     'price': [450, 300, 350, 500, 450, 370, 290]})
book


# Задание 2
# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.
# 

# In[21]:


authors_price = pd.merge(authors, book, on='author_id', how='outer')
authors_price.drop('author_id', axis=1, inplace = True)
authors_price


# Задание 3
# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.
# 

# In[18]:


top5 = authors_price.nlargest(5, 'price')
top5


# In[24]:


authors_price_2 = authors_price.sort_values(by='price', ascending=False)
authors_price_2


# In[25]:


top5_2 = authors_price_2.head(5)
top5_2


# In[27]:


top5_2 == top5


# Задание 4
# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора, минимальная, максимальная и средняя цена на книги этого автора.
# 

# In[90]:


authors_stat = authors_price.copy()
authors_stat


# In[91]:


authors_stat = authors_price.groupby('author_name')
authors_stat.agg(min_price=('price', 'min'), max_price=('price', 'max'), mean_price=('price', 'mean'))


# Задание 5**
# Создайте новый столбец в датафрейме authors_price под названием cover, в нем будут располагаться данные о том, какая обложка у данной книги - твердая или мягкая. В этот столбец поместите данные из следующего списка:
# ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая'].
# Просмотрите документацию по функции pd.pivot_table с помощью вопросительного знака.Для каждого автора посчитайте суммарную стоимость книг в твердой и мягкой обложке. Используйте для этого функцию pd.pivot_table. При этом столбцы должны называться "твердая" и "мягкая", а индексами должны быть фамилии авторов. Пропущенные значения стоимостей заполните нулями, при необходимости загрузите библиотеку Numpy.
# Назовите полученный датасет book_info и сохраните его в формат pickle под названием "book_info.pkl". Затем загрузите из этого файла датафрейм и назовите его book_info2. Удостоверьтесь, что датафреймы book_info и book_info2 идентичны.
# 

# In[93]:


authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price


# In[94]:


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


# In[106]:


book_info = pd.pivot_table(authors_price, values=['price', 'cover'], index=['author_name'], columns=['cover'], 
                           aggfunc={'price': sum})
book_info


# In[107]:


book_info.fillna(0, inplace=True)
book_info


# In[108]:


import pickle


# In[109]:


with open('book_info.pkl', 'wb') as f: 
    pickle.dump(book_info, f)


# In[110]:


with open('book_info.pkl', 'rb') as f: 
    book_info2 = pickle.load(f)


# In[111]:


book_info2 == book_info


# In[112]:


book_info2


# In[ ]:




