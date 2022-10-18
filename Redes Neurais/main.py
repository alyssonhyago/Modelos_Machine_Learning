'''

Redes Neurais Artificiais

Autor: Engenheiro Alysson Hyago

Data: 18/10/2022

'''

# Importação de bibliotecas
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
# Carregando os dados do pré-processamento

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)

X_credit_treinamento.shape, y_credit_treinamento.shape

# Criando o modelo

# 3 -> 2 -> 2 -> #formula para saber o numero de camadas ocultas (Nº de entras + Nº de saídas)/2 -> (3 + 1)/2 = 2
rede_neural_credit = MLPClassifier(max_iter=1000, verbose=True, tol= 0.0000100, solver='adam', activation='relu', hidden_layer_sizes=(2,2)) # com a tol menor, teve melhor resultado
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

# Previsões

previsoes_credit_neural = rede_neural_credit.predict(X_credit_teste)

#Metricas

accuracy_score(y_credit_teste,previsoes_credit_neural)
print(classification_report(y_credit_teste,previsoes_credit_neural))

#Census

#Carregando os dados
file = open('./Bases de dados/census.pkl', 'rb')
X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(file)

X_census_treinamento.shape, y_census_treinamento.shape

#criando o modelo

#(108 + 1)/2 = 55
#108 -> 55 -> 55 -> 1
rede_neural_census = MLPClassifier(max_iter=1000, verbose=True, tol= 0.0000100, solver='adam', activation='relu', hidden_layer_sizes=(55,55))
rede_neural_census.fit(X_census_treinamento, y_census_treinamento)

# Previsões

previsoes_census_neural = rede_neural_census.predict(X_census_teste)

#Metricas

accuracy_score(y_census_teste,previsoes_census_neural)
print(classification_report(y_census_teste,previsoes_census_neural))