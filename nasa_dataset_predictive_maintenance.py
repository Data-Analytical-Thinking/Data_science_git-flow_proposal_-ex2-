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
