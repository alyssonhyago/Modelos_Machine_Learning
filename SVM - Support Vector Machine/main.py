'''

Classificção por instâncias - técnica KNN - k-Nearest Neighbour

Autor: Eng Alysson Hyago Pereira de Olvieira

Data: 16/10/2022

'''


# Importando as bibliotecas
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

#Base de credit

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)

svm_credit = SVC(kernel='rbf', random_state=1, C=2.0)
svm_credit.fit(X_credit_treinamento,y_credit_treinamento)

previsoes_svm_credit = svm_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste,previsoes_svm_credit)
print(classification_report(y_credit_teste,previsoes_svm_credit))

#Base do census

file = open('./Bases de dados/census.pkl', 'rb')
X_censu_treinamento, y_censu_treinamento, X_censu_teste, y_censu_teste = pickle.load(file)

svm_census = SVC(kernel='rbf', random_state=1, C=1.0)
svm_census.fit(X_censu_treinamento,y_censu_treinamento)

previsoes_svm_census = svm_census.predict(X_censu_teste)

accuracy_score(y_censu_teste,previsoes_svm_census)
print(classification_report(y_censu_teste,previsoes_svm_census))