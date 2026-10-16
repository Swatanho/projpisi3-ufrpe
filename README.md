## Tratamento de Dados de Problemas Cardíacos com ML
Projeto de Machine Learning em Python para tratamento de dataset e uso posterior no projeto [Heart Health](https://github.com/Swatanho/projdsi-ufrpe).


Dataset utilizado: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

#### Dependências:
```
pip install -r requirements.txt
```

#### Pastas:
- `data/raw/`: contém os datasets originais (CSVs BRFSS 2015).
- `data/tratados/`: dataset pós-tratamento (`diabetes_processed.parquet`) e demais arquivos úteis de análise.
- `src/`: código-fonte do tratamento de dados e do dashboard.

#### Pipeline para limpeza de dados e exibição da página de análises:

1. Abrir o repositório contendo os códigos em um terminal
2. Executar `py src/data_prep.py` (gera `data/tratados/diabetes_processed.parquet`)
3. Executar `streamlit run app.py` (abre o painel de análises no navegador)

#### Estrutura mínima para execução da limpeza e geração da página de análises
pandas
numpy
scikit-learn
pyarrow
streamlit
plotly
