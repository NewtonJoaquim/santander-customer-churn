#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv


# #### Carregando os dados em um dataframe do pandas

# In[ ]:


train_file = 'data-projeto-santander/train.csv'
data = read_csv(train_file)
print(data.shape)


# #### Checando as colunas e os tipos

# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


data.describe()


# #### Podemos observar alguns problemas: Temos muitas colunas, os dados estão muito esparsos e desbalanceados e temos dados faltantes

# In[ ]:


data.groupby('TARGET').size()


# #### Vamos calcular a correlação entre as colunas

# In[ ]:


data.corr(method = 'pearson')


# In[ ]:


data.skew()


# #### Como temos muitas colunas, fica inviável vizualizar as distribuições com gráficos

# #### Vamos começar o pré-processamento colocando os dados na mesma escala. Utilizaremos o método de Normalização, já que observamos que os dados são esparsos

# In[ ]:


from sklearn.preprocessing import Normalizer


# In[ ]:


data_array = data.values


# In[ ]:


X = data_array[:,0:-1]


# In[ ]:


X


# In[ ]:


Y = data_array[:,-1]


# In[ ]:


Y


# In[ ]:


scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)


# In[ ]:


normalizedX[0:5,:]


# #### Agora precisamos reduzir a dimensionalidade dos dados. Acredito que temos muitas colunas que não agregam muito na predição, já que consistem em muitos valores repetidos. Testarei esta hipótese com o método Ensemble para obter o score de cada atributo

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(normalizedX, Y)


# In[ ]:


print(model.feature_importances_)


# #### Realmente, temos muitas colunas que não são importantes, que possuem score 0 ou muito baixo. 

# In[ ]:


null_score_columns = []
for i in range(len(model.feature_importances_)):
    if model.feature_importances_[i] == 0:
        null_score_columns.append(data.columns[i])
print(null_score_columns)


# #### Antes de realizar a seleção de features e o treinamento do modelo, precisamos fazer a divisão em treino e teste

# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(normalizedX, Y, test_size=0.33)


# #### Iremos fazer uma Seleção de Features para utilizarmos apenas as colunas mais relevantes. Vamos testar com as 20 mais relevantes

# In[68]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(model, 20)
fit = rfe.fit(X_train, Y_train)


# In[ ]:


normalizedX.shape


# In[ ]:


fit.support_


# #### Agora precisamos processar os dados de teste para calculara precisão de nosso modelo

# In[69]:


score = fit.score(X_test, Y_test)
print(score)


# #### Agora, vamos fazer a predição com os dados do arquivo de teste

# In[70]:


test_file = 'data-projeto-santander/test.csv'
test_data = read_csv(test_file)
test_data_array = test_data.values
test_X = test_data_array
test_Y = test_data_array[:,-1]
test_scaler = Normalizer().fit(test_X)
normalized_test_X = test_scaler.transform(test_X)


# In[71]:


normalized_test_X.shape


# In[72]:


def get_relevant_columns(columns, arr):
    new_arr = []
    for i in range(len(columns)):
        if(columns[i] == True):
            new_arr.append(arr[i])
    return new_arr


# In[73]:


get_relevant_columns(fit.support_, normalized_test_X[0])


# In[74]:


rfe_normalized_test_X = []
for i in normalized_test_X:
    rfe_normalized_test_X.append(get_relevant_columns(fit.support_, i))


# In[76]:


prediction = fit.predict(normalized_test_X)


# In[87]:


prediction


# In[88]:


test_data.ID.values


# #### Finalmente, vamos anotar o resultado das predições em um arquivo

# In[91]:


f = open("submission.csv", "w+")
f.write("ID,TARGET\n")
for i in range(len(prediction)):
    f.write(str(test_data.ID.values[i]) + "," + str(int(prediction[i])) + "\n")
f.close()


# In[ ]:




