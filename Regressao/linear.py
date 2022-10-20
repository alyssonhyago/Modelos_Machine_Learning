'''

Regressão linear simples

Autor: Eng Alysson Hyago Pereira de Oliveira

Data: 20/10/2022

'''

#Plano de saúde

#Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px

# Carregando os dados

base_plano_saude = pd.read_csv('./Bases de dados/plano_saude.csv')

X_plano_saude = base_plano_saude.iloc[:, 0].values
y_plano_saude = base_plano_saude.iloc[:, 1].values

np.corrcoef(X_plano_saude,y_plano_saude)

X_plano_saude = X_plano_saude.reshape(-1,1)
X_plano_saude.shape

from sklearn.linear_model import LinearRegression

regressor_plano_saude = LinearRegression()
regressor_plano_saude.fit(X_plano_saude,y_plano_saude)

#bo
regressor_plano_saude.intercept_

#b1
regressor_plano_saude.coef_

#previsoes
previsoes = regressor_plano_saude.predict(X_plano_saude)

grafico = px.scatter(x= X_plano_saude.ravel(), y= y_plano_saude)
grafico.add_scatter(x= X_plano_saude.ravel(), y=previsoes, name= 'Regressão')
grafico.show()

#Métrica
regressor_plano_saude.score(X_plano_saude,y_plano_saude)

#Base das casas

base_casas = pd.read_csv('./Bases de dados/house_prices.csv')

base_casas.describe()

base_casas.isnull().sum()

base_casas.corr()

fig = plt.figure(figsize = (20,20))
sns.heatmap(base_casas.corr(), annot=True)

#Dividindo entre previsores e classes

X_casas = base_casas.iloc[:,5:6].values
y_casas = base_casas.iloc[:,2].values

from sklearn.model_selection import train_test_split

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas,y_casas, random_state=0, test_size=0.20)
X_casas_treinamento.shape, y_casas_treinamento.shape

#treinamento

from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
regressor_casas = LinearRegression()
regressor_casas.fit(X_casas_treinamento,y_casas_treinamento)

regressor_casas.intercept_
regressor_casas.coef_

previsoes = regressor_casas.predict(X_casas_treinamento)

regressor_casas.score(X_casas_treinamento,y_casas_treinamento)

grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y=y_casas_treinamento)
grafico2 = px.line(x=X_casas_treinamento.ravel(), y=previsoes)
grafico3 = go.Figure(data= grafico1.data + grafico2.data)
grafico3.show()