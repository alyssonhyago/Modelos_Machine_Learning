'''

Avaliação de algoritmos de classificação

Autor> Eng. Alysson Hyago Pereira de Oliveira

Data: 18/10/2022

'''
import numpy as np
import pandas as pd

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

#Random forest

parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1,5,10],
              }

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_paramentros)
print(melhor_resultado)

#Knn

parametros = {'n_neighbors': [3,5,10,20],
              'p': [1,2]}

grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_paramentros)
print(melhor_resultado)

#SVM

parametros = {'tol': [0.001, 0.0001, 0.00001],
               'C': [1.0, 1.5, 2.0],
               'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_paramentros)
print(melhor_resultado)


#Redes Neurais



parametros = {'activation': ['relu', 'logistic', 'tahn'],
              'solver':['adam','sgd'],
              'batch_size':[10,56]}

grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_paramentros)
print(melhor_resultado)


#Validação cruzada
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
resultados_arvore = []
resultados_random = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore,X_credit,y_credit, cv=kfold)
    print(scores)
    print(scores.mean())
    resultados_arvore.append(scores.mean())

    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, X_credit, y_credit, cv=kfold)
    resultados_random.append(scores.mean())


#Avaliação dos resultados
len(resultados_arvore)
resultados = pd.DataFrame({'Arvore ': resultados_arvore, 'Random forest': resultados_random})
print(resultados)

resultados.describe() # descrição estatistica observar o std sendo o meor para maior consistencia e ver mean e std melhor

resultados.var() #-> variancia
