## Tratamento de Dados de Problemas Cardíacos com ML
Projeto de Machine Learning em python para tratamento de dataset e uso posterior no projeto [Heart Health](https://github.com/Swatanho/projdsi-ufrpe).


Dataset utilizado: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

#### Dependências:
```
    pip install pandas scikit-learn
```

#### Pastas:
DB: Contém o dataset original

dados_tratados: dataset pós tratamento, com alguns arquivos úteis de análise.

#### Pipeline para limpeza de dados e exibição da página de análises: 

1. Abrir o repositório contendo os códigos em um terminal
2. Executar python tratamento.py
3. Executar python pre_processamento.py
4. Executar python analise_clusters.py
5. Executar python gerar_painel.py
6. Abrir o arquivo painel.html gerado a partir das análises.

#### Estrutura mínima para execução da limpeza e geração da página de análises
pandas
numpy
scikit-learn
