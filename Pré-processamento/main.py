'''

Machine Learning - Pré-processamento
Autor: ALysson Hyago Pereira de Oliveira
Data: 12/10/2022

'''

# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Base de dado de Crédito
## Exploração dos dados

base_credit = pd.read_csv('./Bases de dados/credit_data.csv')

base_credit.head()
base_credit.info()

base_credit.describe()

base_credit[base_credit['loan'] <=1.377630]

#Visualização dos dados

np.unique(base_credit['default'], return_counts=True)
sns.countplot(x = base_credit['default'])

plt.hist(x=base_credit['age'])
plt.hist(x=base_credit['income'])
plt.hist(x=base_credit['loan'])


grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()

# Tratamento de valores inconsistentes

base_credit[base_credit['age'] < 0 ]
base_credit[base_credit['age'] < 0 ].index


## apagar a coluna inteira
base_credit2 = base_credit.drop('age', axis=1)

## apagar os registros com valores inconsistentes

base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

base_credit3[base_credit3['age'] < 0]

## Preencher os valores inconsistentes manualmente



## Preencher com a media

base_credit[base_credit['age'] > 0].mean()
base_credit['age'][base_credit['age'] > 0].mean()

base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92

base_credit.head(30)

## Tratamento de valores faltantes

base_credit.isnull() # varifica valores null 9 vazios
base_credit.isnull().sum()

base_credit.loc[pd.isnull(base_credit['age'])]

base_credit['age'].fillna(base_credit['age'].mean(), inplace= True) # fillna preenche os valores Na

base_credit.loc[pd.isnull(base_credit['age'])]
base_credit.loc[(base_credit['clientid']== 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

base_credit.loc[base_credit['clientid'].isin([29,31,32])] # código mais curto

## Divisão entre previsores e classe

X_credit =  base_credit.iloc[:, 1:4].values # .values muda de pandas data frame para numpy
Y_credit = base_credit.iloc[:, 4].values

## Escalonamento dos atributos

X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min()
X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max()

'''
Obeserva-se pelos codigos acima q existe uma diferença muito grande na escala dos numero, logo devemos padronizar, mesma escala

Padronização: x = (x - média(x))/ std(x) quando existe a presença de outliers

Normalização: x = (x - minimo(x)) / (máximo(x) - minimo(x)) #
'''

from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

## Base de dados do Censo

base_census = pd.read_csv('./Bases de dados/census.csv')
base_census.head()
base_census.info()
base_census.describe()
base_census.isnull().sum()
base_census.isna().sum()

## Visualização da base de dados do census

np.unique(base_census['income'], return_counts=True)
sns.countplot(x = base_census['income'])
plt.hist(x = base_census['age'])

grafico2 = px.treemap(base_census, path=['workclass', 'age'])
grafico2.show()

grafico = px.parallel_categories(base_census,dimensions=['occupation', 'relationship'])
grafico.show()

## Divisão entre previsores e a classe base census

X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values

##Atributos categóricos
### Label enconder -> transforma valores categóricos em dados numéricos
from sklearn.preprocessing import LabelEncoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_martial = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_martial.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

X_census

### One HotEnconder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotenconder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
X_census = onehotenconder_census.fit_transform(X_census).toarray()
X_census.shape

## Escalonamento base census

from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
X_census.shape

### Divisão das bases de treinamento e teste

from sklearn.model_selection import train_test_split

#Credit data
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit,Y_credit, test_size=0.25, random_state=0)
X_credit_treinamento.shape
X_credit_teste.shape

#Cesus

X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census,Y_census, test_size=0.15, random_state=0)

#Salvar as bases de dados

import pickle

with open('credit.pkl', mode= 'wb') as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste], f)

with open('census.pkl', mode= 'wb') as f:
    pickle.dump([X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste], f)