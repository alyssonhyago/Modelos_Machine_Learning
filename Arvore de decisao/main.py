'''

Classificação por técnicas de árvore de decisão

Autor: Eng. Alysson Hyago

Data: 15/10/2022

'''


# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Base risco de crédito

import pickle
file = open('./Bases de dados/risco_credito.pkl', 'rb')
X_risco_credito, y_risco_credito = pickle.load(file)

X_risco_credito.shape, y_risco_credito.shape

arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(X_risco_credito,y_risco_credito)

arvore_risco_credito.feature_importances_

from sklearn import tree

previsores_tree = ['História', 'dívidas', 'garantias', 'rendas']
tree.plot_tree(arvore_risco_credito, feature_names= previsores_tree, class_names= arvore_risco_credito.classes_, filled=True)

# previsões

previsoes = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
previsoes

### Base credit

file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(file)

arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes_credito = arvore_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste,previsoes_credito)
print(classification_report(y_credit_teste,previsoes_credito))

from sklearn import tree
previsores = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows=1,ncols=1,figsize = (20,20))
tree.plot_tree(arvore_credit, feature_names= previsores, class_names= ['0','1'], filled=True)
fig.savefig('arvore_credit.png')