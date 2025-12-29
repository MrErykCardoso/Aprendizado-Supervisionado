#Modelo de classificação de dados comm base no método de decicion trees
#Especificação: classificação de flores usando o dataset Iris

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Carregando os dados
iris = load_iris() #Atribuindo os dados de treinamento a uma variável
X = iris.data #Carregando os atributos (features) em X
y = iris.target #Carrega os objetivos (Labels) em y

# Criando modelo
modelo = DecisionTreeClassifier(max_depth = 3) #Escolhendo modelo de arvore de decisão de classificação com profundidade de 3 nós

# Treinamento
modelo.fit(X, y) #Treinamento do modelo com os dados selecionados

# Teste
exemplo = [[5.0, 3.6, 1.4, 0.2]] #Iris setosa
predicao = modelo.predict(exemplo)
print(f"Predição:\n{iris.target_names[predicao][0]};")

#Visualização da arvore
tree.plot_tree(modelo, feature_names = iris.feature_names, class_names = iris.target_names, filled = True)
plt.show()