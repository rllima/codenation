#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[133]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[141]:


black_friday.columns


# In[142]:


df = black_friday.copy()


# In[143]:


df.info()


# In[144]:


black_friday.shape


# In[17]:


aux = pd.DataFrame({"colunas": df.columns, "tipo": df.dtypes, "missing": df.isna().sum()})


# In[22]:


aux["percent_missing"] = (aux.missing / df.shape[0]) *100


# In[66]:


aux.missing.max()


# In[25]:


df.Age.value_counts()


# In[45]:


type(int(df[df.Gender == 'F'].Age.value_counts()[0]))


# In[50]:


df.User_ID.nunique()


# In[54]:


df.dtypes.nunique()


# In[63]:


null_values = (len(df) - len(df.dropna()))/ len(df)
null_values


# In[125]:


df.Product_Category_3.value_counts().keys()[0]


# In[74]:


df["purchase_norm"] = (df.Purchase - df.Purchase.min())/(df.Purchase.max() - df.Purchase.min())


# In[76]:


df.purchase_norm.mean()


# In[128]:


df["purchase_z"] = (df.Purchase - df.Purchase.mean())/df.Purchase.std()


# In[130]:


len(df[(df.purchase_z > -1) & (df.purchase_z < 1)])


# In[92]:


bool((df['Product_Category_2'].isna() == df['Product_Category_2'].isna()).all())


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[145]:


def q1():
    shape = black_friday.shape
    return shape
    # Retorne aqui o resultado da questão 1.


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[48]:


def q2():
    age_range_f = int(df[df.Gender == 'F'].Age.value_counts()[0])
    return age_range_f


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[51]:


def q3():
    unique_users = df.User_ID.nunique()
    return unique_users


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[56]:


def q4():
    n_types = df.dtypes.nunique()
    # Retorne aqui o resultado da questão 4.
    return n_types


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[64]:


def q5():
    null_values = (len(df) - len(df.dropna()))/ len(df)
    return null_values


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[67]:


def q6():
    max_null_values = aux.missing.max()
    return max_null_values


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[72]:


def q7():
    max_frequent_value = df.Product_Category_3.value_counts().keys()[0]
    return max_frequent_value


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[78]:


def q8():
    new_mean = df.purchase_norm.mean()
    return new_mean


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[131]:


def q9():
    result = len(df[(df.purchase_z > -1) & (df.purchase_z < 1)])
    return result


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[93]:


def q10():
    equals_na = bool((df['Product_Category_2'].isna() == df['Product_Category_2'].isna()).all())
    return equals_na


# In[ ]:




