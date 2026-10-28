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
 
#### Modelo de predicao de DCV (reuso no app HeartHealth):  
1. Executar `py src/train_model.py` (treina o XGBoost e gera `data/modelo/dcv_modelo.pkl` e `data/modelo/metricas_modelo.json`)  
2. Para usar no app HeartHealth, copiar `src/predictor.py` e `data/modelo/dcv_modelo.pkl`:  
 
```python  
from predictor import PredictorDCV  
pred = PredictorDCV.carregar('dcv_modelo.pkl')  
resultado = pred.prever(dicionario_do_paciente)  # dict com as 20 features do contrato  
``` 

3. A aba **"🧪 Simulador de Risco (Modelo)"** do próprio dashboard (`streamlit run app.py`) usa esse modelo no navegador: predição individual com classe de risco e fatores de risco (SHAP)  
