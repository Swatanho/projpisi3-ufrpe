#!/usr/bin/env python3
"""Comparacao de modelos de classificacao para o artigo (Secao 7.2).

Segunda pergunta do projeto: com base nas informacoes coletadas previamente,
o paciente tem (ou desenvolvera) doenca cardiaca?

Gera as Tabelas 1 (treino) e 2 (teste) do artigo, comparando:
  - SVM (RBF)
  - SVM + SMOTE
  - Random Forest (class_weight='balanced')
  - KNN + undersampling
  - Regressao Logistica (class_weight='balanced') - baseline interpretavel
  - XGBoost (scale_pos_weight) - modelo final do projeto

Metricas: precisao, recall, f1-score e suporte por classe; media (macro),
media ponderada, acuracia, alem de ROC-AUC e PR-AUC (globais, mais adequadas
ao desbalanceamento da base).

Decisoes metodologicas:
  - Split estratificado 80/20 (seed 42); teste tocado uma unica vez.
  - Features: as 20 do contrato do app; alvo HeartDiseaseorAttack.
  - StandardScaler apenas para modelos baseados em distancia (SVM, KNN, LR).
  - Sem one-hot: as 20 features ja sao numericas (binarias/ordinais/continuas).
  - Limiar de decisao 0.5 (equivale ao classification_report padrao); o
    threshold operacional 0.36 do XGBoost e discutido a parte no artigo.
  - Por custo computacional dos classificadores classicos (sobretudo SVM e
    KNN), a comparacao roda sobre uma amostra estratificada de SAMPLE_SIZE
    registros, mantendo a proporcao real das duas classes.

Saidas em data/modelo/:
  - tabela_treino.csv / tabela_teste.csv : formato para colar no Google Sheets
  - anexo_auc.csv                        : ROC-AUC e PR-AUC por modelo
  - relatorio_classificacao.json         : relatorio completo
  - resumo_classificacao.md              : tabelas em Markdown p/ o artigo
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'tratados' / 'diabetes_processed.parquet'
OUTPUT_DIR = BASE_DIR / 'data' / 'modelo'

TARGET_COL = 'HeartDiseaseorAttack'

# Contrato de 20 features do app (DSI) - identico ao clustering e ao treino.
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
]

SEED = 42
TEST_SIZE = 0.20
SAMPLE_SIZE = 10_000  # amostra estratificada p/ os modelos classicos (SVM/KNN)


def carregar_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Arquivo tratado nao encontrado em: {DATA_PATH}. Execute src/data_prep.py.'
        )
    df = pd.read_parquet(DATA_PATH)
    faltando = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if faltando:
        raise ValueError(f'Colunas ausentes na base tratada: {faltando}')
    return df


def construir_modelos(scale_pos_weight: float) -> dict:
    """Retorna {nome: estimador}. Estimadores com predict/predict_proba."""
    return {
        'SVM': make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', class_weight='balanced', random_state=SEED),
        ),
        'SVM + SMOTE': make_imb_pipeline(
            StandardScaler(),
            SMOTE(random_state=SEED),
            SVC(kernel='rbf', random_state=SEED),
        ),
        'RF': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
        'KNN + Undersampling': make_imb_pipeline(
            StandardScaler(),
            RandomUnderSampler(random_state=SEED),
            KNeighborsClassifier(n_neighbors=5),
        ),
        'Reg. Logistica': make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
        ),
        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=1000,
            early_stopping_rounds=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def _classe(rep: dict, classe: str, chave: str) -> float:
    """Le um valor do classification_report tolerando classe ausente."""
    return rep.get(classe, {}).get(chave, 0.0)


def _escore(est, X) -> np.ndarray:
    """Score continuo para AUC (probabilidade ou decision_function).

    SVM sem `probability=True` expoem apenas decision_function; os demais
    expoem predict_proba. Ambos servem para ROC/PR-AUC (ordenação).
    """
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def avaliar(est, X, y) -> tuple:
    """Retorna (classification_report, roc_auc, pr_auc) no threshold 0.5."""
    pred = est.predict(X)
    escore = _escore(est, X)
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    roc = roc_auc_score(y, escore)
    pr = average_precision_score(y, escore)
    return rep, roc, pr


def montar_tabela(resultados: dict) -> pd.DataFrame:
    """Transforma a dict {modelo: (rep, roc, pr)} no formato do template.

    Linha por (modelo, metrica): classes 0 e 1, media, media ponderada,
    acuracia e (extra) ROC-AUC e PR-AUC.
    """
    linhas = []
    for modelo, (rep, roc, pr) in resultados.items():
        for metrica in ['precision', 'recall', 'f1-score']:
            rotulo = {'precision': 'Precisao', 'recall': 'Recall', 'f1-score': 'F1-Score'}[metrica]
            linhas.append({
                'Modelo': modelo,
                'Metrica': rotulo,
                'Classe 0': _classe(rep, '0', metrica),
                'Classe 1': _classe(rep, '1', metrica),
                'Media': rep['macro avg'][metrica],
                'Media Ponderada': rep['weighted avg'][metrica],
                'Acurácia': '',  # preenchida so na linha de acuracia / suporte
                'ROC-AUC': roc,
                'PR-AUC': pr,
            })
        linhas.append({
            'Modelo': modelo,
            'Metrica': 'Suporte',
            'Classe 0': rep.get('0', {}).get('support', 0.0),
            'Classe 1': rep.get('1', {}).get('support', 0.0),
            'Media': '',
            'Media Ponderada': '',
            'Acurácia': '',
            'ROC-AUC': roc,
            'PR-AUC': pr,
        })
        linhas.append({
            'Modelo': modelo,
            'Metrica': 'Acurácia',
            'Classe 0': '',
            'Classe 1': '',
            'Media': '',
            'Media Ponderada': '',
            'Acurácia': rep['accuracy'],
            'ROC-AUC': roc,
            'PR-AUC': pr,
        })
    return pd.DataFrame(linhas)


def executar() -> dict:
    df = carregar_base()
    X = df[FEATURES].astype(float)
    y = df[TARGET_COL].astype(int)

    print('=' * 66)
    print(' COMPARACAO DE MODELOS DE CLASSIFICACAO (Secao 7.2)')
    print('=' * 66)
    print(f'Base tratada: {len(df):,} registros | prevalencia {y.mean():.3f}')
    print(f'Amostra estratificada p/ modelos classicos: {SAMPLE_SIZE:,}')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )

    # Amostra estratificada para viabilizar SVM/KNN (custo quadratico)
    rs = np.random.RandomState(SEED)
    idx = []
    for classe in (0, 1):
        pos = np.where(y_tr == classe)[0]
        n = max(1, int(SAMPLE_SIZE * len(pos) / len(y_tr)))
        idx.extend(rs.choice(pos, size=n, replace=False))
    idx = np.array(idx)
    X_am, y_am = X_tr.iloc[idx], y_tr.iloc[idx]
    print(f'Amostra de treino: {len(X_am):,} | teste: {len(X_te):,}')

    spw = float((y_am == 0).sum() / max((y_am == 1).sum(), 1))
    modelos = construir_modelos(scale_pos_weight=spw)

    resultados = {}
    for nome, est in modelos.items():
        print(f'-> treinando: {nome}')
        if nome == 'XGBoost':
            # Early stopping com validacao interna (mesma filosofia do
            # train_model.py): 10% do treino, estratificado.
            X_in, X_val, y_in, y_val = train_test_split(
                X_am, y_am, test_size=0.10, stratify=y_am, random_state=SEED,
            )
            est.fit(
                X_in, y_in,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            est.fit(X_am, y_am)
        rep_tr, roc_tr, pr_tr = avaliar(est, X_am, y_am)
        rep_te, roc_te, pr_te = avaliar(est, X_te, y_te)
        resultados[nome] = (
            rep_tr, roc_tr, pr_tr, rep_te, roc_te, pr_te,
        )

    # Tabelas treino/teste
    treino = montar_tabela({m: (r[0], r[1], r[2]) for m, r in resultados.items()})
    teste = montar_tabela({m: (r[3], r[4], r[5]) for m, r in resultados.items()})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    treino.to_csv(OUTPUT_DIR / 'tabela_treino.csv', index=False)
    teste.to_csv(OUTPUT_DIR / 'tabela_teste.csv', index=False)

    # Anexo AUC (globais, adequadas ao desbalanceamento)
    anexo = pd.DataFrame([
        {
            'Modelo': nome,
            'ROC-AUC treino': r[1],
            'PR-AUC treino': r[2],
            'ROC-AUC teste': r[4],
            'PR-AUC teste': r[5],
        }
        for nome, r in resultados.items()
    ])
    anexo.to_csv(OUTPUT_DIR / 'anexo_auc.csv', index=False)

    # Relatorio JSON completo
    relatorio = {
        'features': FEATURES,
        'seed': SEED,
        'test_size': TEST_SIZE,
        'sample_size_treino_classicos': int(len(X_am)),
        'prevalencia_base': float(y.mean()),
        'aplicada_no_treinoo': {
            m: {'roc_auc': r[1], 'pr_auc': r[2]}
            for m, r in resultados.items()
        },
        'aplicada_no_teste': {
            m: {'roc_auc': r[4], 'pr_auc': r[5]}
            for m, r in resultados.items()
        },
    }
    (OUTPUT_DIR / 'relatorio_classificacao.json').write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=float),
        encoding='utf-8',
    )

    # Resumo Markdown para o artigo
    md = []
    md.append('# Resultados da classificação (Seção 7.2)\n')
    md.append(f'> Base: {len(df):,} registros | alvo `HeartDiseaseorAttack` '
              f'(prevalência {y.mean():.1%}). Split 80/20 estratificado (seed {SEED}).\n')
    md.append('\n## Tabela 1 — Desempenho no TREINO (amostra estratificada)\n')
    md.append(treino.to_markdown(index=False))
    md.append('\n## Tabela 2 — Desempenho no TESTE (hold-out 20%)\n')
    md.append(teste.to_markdown(index=False))
    md.append('\n## Anexo — ROC-AUC e PR-AUC (globais)\n')
    md.append(anexo.to_markdown(index=False, floatfmt='.3f'))
    (OUTPUT_DIR / 'resumo_classificacao.md').write_text(
        '\n'.join(md), encoding='utf-8',
    )

    print('\nTabela 1 (treino):')
    print(treino.to_string(index=False))
    print('\nTabela 2 (teste):')
    print(teste.to_string(index=False))
    print('\nAnexo AUC:')
    print(anexo.round(3).to_string(index=False))
    print(f'\nArquivos salvos em {OUTPUT_DIR}')
    return treino, teste, anexo


if __name__ == '__main__':
    executar()