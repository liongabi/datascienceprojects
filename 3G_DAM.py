#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Autora @gabriele.leao
#código com objetivo de manusear o dataset dos níveis operacionais e descargas defluentes de Três Gargantas, China. 

#Dataset disponível em: https://www.kaggle.com/konivat/three-gorges-dam-water-data

#playlist: https://open.spotify.com/playlist/37i9dQZF1DWTggY0yqBxES?si=3850314a85824856


# In[87]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import math
import io


# In[88]:


dt = pd.read_csv("C:/Users/gabri/OneDrive/Área de Trabalho/Projects/P1 - China DAM/threegorges-water-storage.csv")


# In[89]:


dt.head()


# In[90]:


dt['measurement_date']=pd.to_datetime(dt['measurement_date']) #convertendo coluna de data em datetime

dt.info() #informações sobre meu dataset


# In[91]:


dt.isna().sum() #missing values/columns


# In[92]:


dt.describe() #estatísticas gerais do dataset


# In[ ]:


#estatisticas gerais


# In[113]:


#criando coluna de anos

dt['year']=dt['measurement_date'].dt.year
dt['month']=dt['measurement_date'].dt.month


# In[114]:


dt.head()


# In[120]:


#timeseries

fig, ax = plt.subplots(2,2, figsize=(8, 8), dpi= 600)
                       
plt.subplot(2, 2, 1)
aa=fig.add_subplot(221);
sns.lineplot(data=dt, x="year", y="upstream_water_level", ax=aa)
plt.xticks(rotation=45)
aa.set_title('Upstream water level')
aa.set_ylabel('Level')
aa.set_xlabel(None)


plt.subplot(2, 2, 2)
ab=fig.add_subplot(222);
sns.lineplot(data=dt, x="year", y="downstream_water_level", ax=ab)
plt.xticks(rotation=45)
ab.set_title('Downstream water level')
ab.set_ylabel('Level')
ab.set_xlabel(None)

plt.subplot(2, 2, 3)
ac=fig.add_subplot(223);
sns.lineplot(data=dt, x="year", y="inflow_rate", ax=ac)
ac.set_title('Inflow')
ac.set_ylabel('Discharge')
plt.xticks(rotation=45)
ac.set_xlabel(None)

plt.subplot(2, 2, 4)
ad=fig.add_subplot(224);
sns.lineplot(data=dt, x="year", y="outflow_rate", ax=ad)
ad.set_title('Outflow')
ad.set_ylabel('Discharge')
plt.xticks(rotation=45)
ad.set_xlabel(None)

fig.tight_layout()


# In[50]:


# outflow x nivel jusante

fig=plt.figure()
ax=fig.add_subplot(111);
ax.plot(dt['outflow_rate'], dt['downstream_water_level'], 'o') 
ax.set_title('Outflow x Downstream water level')
ax.set_xlabel('Outflow')
ax.set_ylabel('Donwstream water level')

