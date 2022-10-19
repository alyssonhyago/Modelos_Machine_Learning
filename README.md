# Modelos_Machine_Learning
Modelos de Machine Learning com Linguagem Python

Neste repositório estam presentes Modelos Supervisionados de Classificação e Regressão utilizando técnicas de Machine Learning como:<br>
* Classificação:
  * Naive Bayes
  * Árvore de Decisão
  * KNN 
  * SVM
  * Redes Neurais
  * Regressão linear, multipla, polinomial
  * Árvore de Decisão regressora
  * SVR 
* Regressão
  * Regressão Linear
  * Regressão Multipla
  * Regressão Polinomial
---

## Avavaliação dos algoritmos

Utiliza-se conceitos de estatisticas descritiva para avaliar os algortimos e também se faz uso de técnias de tunning para obter os melhores parãmetros paara os modelos.

* Girdsearch
* Cross validation
* Variância
* Desvio padrão
* Teste de hipótese
  * Hipótese nula(H0)
  * Hipótese alternativa(h1)
    * Alpha -> 0.01 ou 0.05, probabilidade de rejeitar a hipótese nula.
    * p-value >= alpha: não rejeita H0
    * p-value < alpha: rejeita H0
* Teste de hipótese Z (distribuição normal)
  * Calcula o valor de Z , busca a tabela padrão
  * valor de p = 1 - Z 
  * Se o valor da média é maior que a média atual, esta do lado direito da distribuição normal, logo tem q ser subtraido de 1.
  * Caso contrário, não precisa fazer a subtração.
  
* Teste de normalidade
  * hipotese nulo significa que os dados estao na distribuição normal, ou seja , p >= 0.05
  * teste de Shapiro nos resultados dos modelos
  * utilizar a biblioteca seaborn para ver as distribuições graficamente função displot()

* Teste Anova
  * Comparação entre 3 ou mais grupos ( amostras independentes)
  * Distribuição normal
  * Variação entre grupos comparando a variação dentro dos grupos
  * H0 : não há diferença estatística
  * H1: existe diferença estatística
  * objetivo é provar que existe diferença estatística
  * Tabela F = (SSG/DFG)/(SSE/DFE)
  * F crítico na tabela
    *  entre 0 e F critico : não existe diferença estatistica
    * Superior a F crítico : existe diferença estatística
  
