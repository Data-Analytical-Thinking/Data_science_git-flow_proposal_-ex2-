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

