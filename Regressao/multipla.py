'''

Regressão linear multipla

Autor: Eng Alysson Hyago Pereira de Oliveira

Data: 20/10/2022

'''


#Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from sklearn.linear_model import LinearRegression

base_casas = pd.read_csv('./Bases de dados/house_prices.csv')
base_casas.head()

X_casas = base_casas.iloc[:,3:19].values
y_casas = base_casas.iloc[:,2].values

from sklearn.model_selection import train_test_split

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas,y_casas, random_state=0, test_size=0.30)
X_casas_treinamento.shape, y_casas_teste.shape

regressor_multiplos_casas = LinearRegression()
regressor_multiplos_casas.fit(X_casas_treinamento, y_casas_treinamento)

regressor_multiplos_casas.intercept_
regressor_multiplos_casas.coef_

regressor_multiplos_casas.score(X_casas_treinamento, y_casas_treinamento)
regressor_multiplos_casas.score(X_casas_teste, y_casas_teste)


previsores_multiplos = regressor_multiplos_casas.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_casas_teste,previsores_multiplos)
mean_squared_error(y_casas_teste,previsores_multiplos)