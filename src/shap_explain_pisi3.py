#!/usr/bin/env python3
"""Explicabilidade SHAP sobre o modelo FINAL (XGBoost) - Secao 7.2.

Usa o artefato ja treinado em data/modelo/dcv_modelo.pkl (produzido por
src/train_model.py). Gera, sobre uma amostra estratificada do teste:
  - importancia global por magnitude media do SHAP (sumario em CSV);
  - valores SHAP medios por feature (bar), salvos em CSV;

Observacao: o pacote externo `shap` nao e necessario - usamos as
contribuicoes nativas do XGBoost (predictor + pred_contribs).
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.predictor import PredictorDCV
except ModuleNotFoundError:  # execucao direta do arquivo
    from predictor import PredictorDCV

BASE_DIR = Path(__file__).resolve().parent.parent
ARTEFATO = BASE_DIR / 'data' / 'modelo' / 'dcv_modelo.pkl'
DATA_PATH = BASE_DIR / 'data' / 'tratados' / 'diabetes_processed.parquet'
OUTPUT_DIR = BASE_DIR / 'data' / 'modelo'

TARGET_COL = 'HeartDiseaseorAttack'
SEED = 42
N_AMOSTRA = 2000


def executar():
    pred = PredictorDCV.carregar(ARTEFATO)
    df = pd.read_parquet(DATA_PATH)

    # Amostra estratificada (proporção real das classes)
    rs = np.random.RandomState(SEED)
    idx = []
    for classe in (0, 1):
        pos = np.where(df[TARGET_COL].astype(int) == classe)[0]
        n = max(1, int(N_AMOSTRA * len(pos) / len(df)))
        idx.extend(rs.choice(pos, size=n, replace=False))
    amostra = df.iloc[idx].reset_index(drop=True)
    X = amostra[pred.features].astype(float)

    # Contribuicoes SHAP via pred_contribs nativo
    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=pred.features)
    it = getattr(pred._modelo, 'best_iteration', None)
    if it is None:
        contribs = pred._modelo.predict(dm, pred_contribs=True)
    else:
        contribs = pred._modelo.predict(dm, pred_contribs=True, iteration_range=(0, it + 1))
    shap_values = contribs[:, :-1]  # ultima coluna e o bias

    # Importancia global = media do valor absoluto do SHAP
    importancia = pd.DataFrame({
        'feature': pred.features,
        'shap_media_absoluta': np.abs(shap_values).mean(axis=0),
        'shap_media': shap_values.mean(axis=0),
    }).sort_values('shap_media_absoluta', ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importancia.to_csv(OUTPUT_DIR / 'shap_importancia.csv', index=False)
    np.save(OUTPUT_DIR / 'shap_valores.npy', shap_values)
    np.save(OUTPUT_DIR / 'shap_X.npy', X.to_numpy())  # valores das features (p/ beeswarm)

    print('=' * 60)
    print(' IMPORTANCIA GLOBAL (SHAP) - modelo final XGBoost')
    print('=' * 60)
    print(importancia.sort_values('shap_media_absoluta', ascending=False).to_string(index=False))
    print(f'\nAmostra usada: {len(amostra):,} registros')
    print(f'Arquivos salvos em {OUTPUT_DIR}')

    return importancia


if __name__ == '__main__':
    executar()