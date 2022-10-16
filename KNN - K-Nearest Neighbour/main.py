'''

Classificção por instâncias - técnica KNN - k-Nearest Neighbour

Autor: Eng Alysson Hyago Pereira de Olvieira

Data: 16/10/2022
'''

#Importando as bibliotecas
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

#Base credit

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)



knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes_knn_credit = knn_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste,previsoes_knn_credit)
print(classification_report(y_credit_teste,previsoes_knn_credit))

#Base census

file = open('./Bases de dados/census.pkl', 'rb')
X_censu_treinamento, y_censu_treinamento, X_censu_teste, y_censu_teste = pickle.load(file)

knn_census = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
knn_census.fit(X_censu_treinamento,y_censu_treinamento)

previsoes_knn_census = knn_census.predict(X_censu_teste)

accuracy_score(y_censu_teste,previsoes_knn_census)
print(classification_report(y_censu_teste,previsoes_knn_census))