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

#Teste de normalidade nos resultados


#hipotese nulo significa que os dados estao na distribuição normal, ou seja , p >= 0.05
import seaborn as sns
from scipy.stats import shapiro
alpha = 0.05
shapiro(resultados_arvore)
shapiro(resultados_random)

sns.displot(resultados_arvore,kind='kde')
sns.displot(resultados_random, kind='kde')

#Teste ANOVA e Tukey

from scipy.stats import f_oneway

_, p = f_oneway(resultados_arvore, resultados_random)
print(p)

alpha = 0.05

if p<= alpha:
    print('Hipotese nula rejeitada. Dados são diferentes')
else:
    print('HIpotese alternativa rejeitada. resultados são iguais') # tanto faz usar qualquer um. caso fosse esse.

# Salvando um modelo
#### Carregando so dados do pré-processamento

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)

X_credit_treinamento.shape, y_credit_treinamento.shape

# concatenar as duas variaveis de treinamento e teste

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)

X_credit.shape

y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

y_credit.shape

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

classificador_rede_neural = MLPClassifier(activation='relu', batch_size = 56, solver='adam')
classificador_rede_neural.fit(X_credit, y_credit)

classificador_arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
classificador_arvore.fit(X_credit, y_credit)

classificador_svm = SVC(C = 2.0, kernel='rbf', probability=True)
classificador_svm.fit(X_credit, y_credit)

import pickle
pickle.dump(classificador_rede_neural, open('rede_neural_finalizado.sav', 'wb'))
pickle.dump(classificador_arvore, open('arvore_finalizado.sav', 'wb'))
pickle.dump(classificador_svm, open('svm_finalizado.sav', 'wb'))

#Carregar um classificdor já treinado

rede_neural = pickle.load(open('./Modelos salvos/rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('./Modelos salvos/arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('./Modelos salvos/svm_finalizado.sav', 'rb'))

novo_registro = X_credit[0]
novo_registro = novo_registro.reshape(1,-1)
novo_registro.shape

rede_neural.predict(novo_registro)
arvore.predict(novo_registro)
svm.predict(novo_registro)