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
