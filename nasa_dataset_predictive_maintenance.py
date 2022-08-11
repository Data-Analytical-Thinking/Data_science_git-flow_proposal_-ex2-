import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# definição do caminho dos arquivos para ler os dados
caminho_diretorio = 'https://raw.githubusercontent.com/andressaapio/nasa_dataset/main/CMAPSSData/' 

# definição dos nomes das colunas para os índices
indices_lista = ['motor', 'ciclo_tempo']
configuracao_lista = ['config_1', 'config_2', 'config_3']

# para não precisar escrever o nome dos 21 sensores, vamos utilizar o loop for
sensores_lista = []
for n in range(1, 22):
    sensores_lista.append(f'sensor_{n}')

colunas = indices_lista + configuracao_lista + sensores_lista

treino = pd.read_csv(caminho_diretorio+'train_FD001.txt', sep='\s+', header=None, names=colunas)
teste = pd.read_csv(caminho_diretorio+'test_FD001.txt', sep='\s+', header=None, names=colunas)
y_teste = pd.read_csv(caminho_diretorio+'RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

treino.head()

treino.tail()

treino.mean()

treino['sensor_2']

treino[treino['motor'] == 1]

treino[treino['motor'] == 1]['sensor_2'].plot();

#Parte 2: Análise dos dados

treino.describe()

treino[sensores_lista].describe().transpose()

plt.figure(figsize=(15,10))
corrmat = treino.corr()
sns.heatmap(corrmat, cmap="RdBu_r");

cols = treino.columns
n_cols = min(len(cols), 5)
n_rows = int(np.ceil(len(cols) / n_cols))

sns.set()
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,12))
axes = axes.flatten()
for col, ax in zip(cols, axes):
    ax=sns.histplot(treino[col], kde=True, ax=ax, label=col)
    ax.legend(loc=1)

sns.boxplot(y='sensor_14', data=treino);

sensor = ['sensor_14']
Q1 = treino[sensor].quantile(0.25)
Q3 = treino[sensor].quantile(0.75)
IQR = Q3 - Q1

treino_sem_outliers = treino[~((treino[sensor] < (Q1 - 1.5 * IQR)) | (treino[sensor] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.hist([treino[sensor], treino_sem_outliers[sensor]], bins=20, label=['com outliers', 'sem outliers'])
plt.legend()
plt.show();

treino[['motor', 'ciclo_tempo']]

treino.groupby('motor')['ciclo_tempo'].transform(max)

treino['RUL'] = treino.groupby('motor')['ciclo_tempo'].transform(max) - treino['ciclo_tempo']

treino[treino['motor'] == 1]['RUL'].tail(20)

#Parte 3: Machine Learning

lista_sensores_remover = [1, 5, 6, 9, 10, 16, 18, 19]
remover_sensores = ['sensor_' + str(item) for item in lista_sensores_remover]

remover_colunas = indices_lista + configuracao_lista + remover_sensores

X_treino = treino.drop(remover_colunas, axis=1) 

y_treino = X_treino.pop('RUL') 

X_teste = teste.groupby('motor').last().reset_index().drop(remover_colunas, axis=1)

X_teste

# criação de função de avaliação dos modelos
def avaliacao(y_real, y_pred, legenda):
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f'conjunto {legenda} MAE:{mae}, R2:{r2}')

# ajuste do modelo de árvores aleatórias com os dados de treino
rf = RandomForestRegressor(random_state=10)
rf.fit(X_treino, y_treino)

# predição e avaliação com os dados de treino para o modelo gerado
y_pred_treino = rf.predict(X_treino)
avaliacao(y_treino, y_pred_treino, 'treino')

# predição e avaliação com os dados de teste para o modelo gerado
y_pred_teste = rf.predict(X_teste)
avaliacao(y_teste, y_pred_teste, 'teste')

# comparação gráfica do tempo de vida útil remanescente (RUL) real e predito pelo modelo
x = np.arange(0,100)
plt.figure(figsize=(15,4))
plt.bar(x, y_pred_teste, label='predito')
plt.bar(x, y_teste['RUL'], label='real', alpha=0.7)
plt.legend();

y_treino.plot()
#y_teste.plot()

# Relação entre valores preditos e reais
sns.scatterplot(x=y_teste['RUL'], y=y_pred_teste)
plt.xlabel('Real')
plt.ylabel('Predito')

ind_menor_150 = y_treino < 150

y_treino_150 = y_treino[ind_menor_150]
X_treino_150 = X_treino[ind_menor_150]

X_treino_150

# predição e avaliação com os dados de treino para o modelo gerado
y_pred_treino = rf.predict(X_treino_150)
avaliacao(y_treino_150, y_pred_treino, 'treino')

# predição e avaliação com os dados de teste para o modelo gerado
y_pred_test_150 = rf.predict(X_teste)
avaliacao(y_teste, y_pred_test_150, 'teste')

# comparação gráfica do tempo de vida útil remanescente (RUL) real e predito pelo modelo
plt.figure(figsize=(15,4))
plt.bar(x, y_pred_test_150, label='predito')
plt.bar(x, y_teste['RUL'], label='real', alpha=0.7)
plt.legend();

sns.scatterplot(x=y_teste['RUL'], y=y_pred_test_150)
plt.xlabel('Real')
plt.ylabel('Predito')