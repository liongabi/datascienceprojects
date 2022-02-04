#!/usr/bin/env python
# coding: utf-8

# In[164]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


# In[2]:


"""

[0] age
[1] sex (1 = male; 0 = female)
[2] chest pain type (4 values)
[3] resting blood pressure (in mm Hg on admission to the hospital)
[4] serum cholestoral in mg/dl
[5] fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
[6] resting electrocardiographic results (values 0,1,2)
[7] maximum heart rate achieved
[8] exercise induced angina (1 = yes; 0 = no)
[9] oldpeak = ST depression induced by exercise relative to rest
[10] the slope of the peak exercise ST segment
[11] number of major vessels (0-3) colored by flourosopy
[12] thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

[13] target (1= true; 0 =false)

The "goal" field refers to the presence of heart disease in the patient. 
It is integer valued from 0 (no presence) to 4.


"""


# In[165]:


dt= pd.read_csv(r'C:\Users\gabri\OneDrive\Área de Trabalho\Projects\Heart Disease UCI\heart.csv')


# In[166]:


dt


# In[217]:


dt['sex'].sum()

m=207/302 *100
m


# In[167]:


dt.isnull().sum() #sem dados faltantes


# In[168]:


dt.info()


# In[169]:


sns.histplot(x=dt['age'], palette="rocket") #distribuição aparentemente normal da idade


# In[170]:


sns.histplot(x=dt['thalach'], palette="rocket") 


# In[193]:


sns.histplot(x=dt['chol'], palette="rocket") #base de dados com tamanho ok em cada amostra


# In[172]:


sns.set_theme(style='ticks')

sns.countplot(y=dt['sex'], hue=dt['target'], data=dt, palette="rocket")
#y.tic


# In[213]:


fig, axs = plt.subplots(1,2)
fig.suptitle('Data analysis')

fig.show()

plt.subplot(2, 2, 1)
sns.histplot(x=dt['age']).set(xlabel='Age', ylabel='Total')

plt.subplot(2, 2, 2)
sns.histplot(x=dt['chol']).set(xlabel='Cholesterol mg/L', ylabel='Total')

plt.subplot(2, 2, 3)
sns.countplot(y=dt['cp'], hue=dt['target'], data=dt, palette="Paired").set(xlabel='Total', ylabel='Chest pain')
plt.yticks([0,1,2,3],[1,2,3,4])

plt.subplot(2, 2, 4)
sns.countplot(y=dt['sex'], hue=dt['target'], data=dt,palette="Paired").set(xlabel='Total', ylabel=None)
plt.yticks([1,0],['Male', 'Female'])


fig.tight_layout()


# In[12]:


grafico= px.parallel_categories(dt, dimensions=['age', 'sex'])
grafico.show()


# In[ ]:





# In[ ]:


##     APLICAÇÃO DE DIFERENTES ALGORITMOS PARA PREVISÃO E DECIDIR QUAL É O MELHOR     ##

"""

não usados inicialmente 
[3] resting blood pressure (in mm Hg on admission to the hospital)
[4] serum cholestoral in mg/dl
[5] fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
[6] resting electrocardiographic results (values 0,1,2)
[7] maximum heart rate achieved
[8] exercise induced angina (1 = yes; 0 = no)
[9] oldpeak = ST depression induced by exercise relative to rest
[10] the slope of the peak exercise ST segment


previsores: x=[:, 0:12]
[0] age
[1] sex (1 = male; 0 = female)
[2] chest pain type (4 values)
[11] number of major vessels (0-3) colored by flourosopy
[12] thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

classe: y=[:, 13]
[13] target (1= true; 0 =false)

"""


# In[174]:


x_heart= dt.iloc[:, 0:13].values
y_heart= dt.iloc[:, 13].values

x_heart.shape,y_heart.shape


# In[31]:


#Encoder

"""from sklearn.preprocessing import LabelEncoder
encoder0=LabelEncoder()
encoder1=LabelEncoder()
encoder2=LabelEncoder()
encoder3=LabelEncoder()
encoder4=LabelEncoder()
encoder5=LabelEncoder()
encoder6=LabelEncoder()
encoder7=LabelEncoder()
encoder8=LabelEncoder()
encoder9=LabelEncoder()
encoder10= LabelEncoder()
encoder11= LabelEncoder()
encoder12= LabelEncoder()
encoder13= LabelEncoder()

x_heart[:,0]= encoder0.fit_transform(x_heart[:,0])
x_heart[:,1]= encoder1.fit_transform(x_heart[:,1])
x_heart[:,2]= encoder2.fit_transform(x_heart[:,2])
x_heart[:,3]= encoder3.fit_transform(x_heart[:,3])
x_heart[:,4]= encoder4.fit_transform(x_heart[:,4])
x_heart[:,5]= encoder5.fit_transform(x_heart[:,5])
x_heart[:,6]= encoder6.fit_transform(x_heart[:,6])
x_heart[:,7]= encoder7.fit_transform(x_heart[:,7])
x_heart[:,8]= encoder8.fit_transform(x_heart[:,8])
x_heart[:,9]= encoder9.fit_transform(x_heart[:,9])
x_heart[:,10]= encoder10.fit_transform(x_heart[:,10])
x_heart[:,11]= encoder11.fit_transform(x_heart[:,11])
x_heart[:,12]= encoder12.fit_transform(x_heart[:,12])
"x_heart[:,13]= encoder13.fit_transform(x_heart[:,13])"

"""


# In[175]:


x_heart


# In[176]:


#Escalonamento

from sklearn.preprocessing import StandardScaler

heart_scaler=StandardScaler()
x_heart_scaler= heart_scaler.fit_transform(x_heart)
#y_heart_scaler= heart_scaler.fit_transform(y_heart.reshape(-1,1))


# In[177]:


x_heart_scaler


# In[178]:


#BASE TREINAMENTO E TESTE

from sklearn.model_selection import train_test_split
xh_train, xh_test, yh_train, yh_test= train_test_split(x_heart_scaler, y_heart, test_size=0.15, random_state=0)


# In[179]:


xh_test.shape, xh_train.shape


# In[180]:


yh_test.shape, yh_train.shape


# In[ ]:


"""

Classificadores testados e resultados (accuracy) obtidos:

1. KNEIGHBORS CLASSIFIER: 86,99%
2. SVC: 86,99%
3. ENSEMBLE CLASSIFIERS - random forest: 84,78%
 
"""


# In[181]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


# In[182]:


from sklearn.neighbors import KNeighborsClassifier

knn_heart= KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn_heart.fit(xh_train, yh_train)
prev_knn= knn_heart.predict(xh_test)
accuracy_score(yh_test, prev_knn)


# In[183]:


from sklearn import svm

svm_heart= svm.SVC(kernel='rbf', random_state=0, C=1.5)
svm_heart.fit(xh_train, yh_train)

prev_svm= svm_heart.predict(xh_test)
accuracy_score(yh_test, prev_svm)


# In[184]:


from sklearn.ensemble import RandomForestClassifier

rf_heart= RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
rf_heart.fit(xh_train, yh_train)
prev_rf= rf_heart.predict(xh_test)
accuracy_score(yh_test, prev_rf)


# In[ ]:




