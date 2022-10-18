'''

Avaliação de algoritmos de classificação

Autor> Eng. Alysson Hyago Pereira de Oliveira

Data: 18/10/2022

'''
import numpy as np

'''
Avaliação dos algortimos obtidos

 * Naive bayes: 93.80
 * Árvore de decisão: 98.20
 * Random forest: 98.40
 * Knn: 98.60
 * SVM 98.80
 * Redes neurais: 99.60

'''

# Tuning dos paramentros com GridSearch

## Preparação dos dados
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

#### Carregando so dados do pré-processamento

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)

X_credit_treinamento.shape, y_credit_treinamento.shape

# concatenar as duas variaveis de treinamento e teste

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)

X_credit.shape

y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

y_credit.shape

#Árvore de Decisão

parametros = {'criterion': ['gini', 'entropy'],
              'splitter': ['best','random'],
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,5,10],
              }

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_paramentros)
print(melhor_resultado)
