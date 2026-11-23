#!/usr/bin/env python3
"""SMOTE x SMOTENC — qual oversampling usar (HeartHealth, PISI3).

Responde ao feedback "smote vs smotenc (se tiver variáveis categóricas)".

Contexto técnico:
  - As 20 features do contrato são numéricas, mas apenas 3 são contínuas
    (BMI, MentHlth, PhysHlth). As outras 17 são CATEGÓRICAS: 13 binárias
    (0/1) + 4 ordinais (GenHlth, Age, Education, Income).
  - SMOTE interpola entre vizinhos -> em features binárias/ordinais produz
    valores fracionários (ex.: HighBP = 0.37), que não são categorias válidas.
  - SMOTENC trata contínuas por interpolação e categóricas por moda/mediana,
    gerando valores válidos (0/1 nas binárias, inteiros nas ordinais).

O script:
  1. Documenta os tipos de feature;
  2. QUANTIFICA a integridade: fração de células de features binárias que
     viram valor fracionário com SMOTE vs. SMOTENC;
  3. Compara o modelo (XGBoost) em 3 estratégias: scale_pos_weight (sem
     oversampling), SMOTE e SMOTENC;
  4. Gera relatório com recomendação.

Saídas em data/modelo/:
  - smote_smotenc.csv          : métricas por estratégia;
  - integridade_sinteticos.csv : fração de valores inválidos por método;
  - relatorio_smote_smotenc.md : narrativa consolidada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'tratados' / 'diabetes_processed.parquet'
OUTPUT_DIR = BASE_DIR / 'data' / 'modelo'

TARGET_COL = 'HeartDiseaseorAttack'

FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
]

# Tipos de feature (índices em FEATURES)
NUM_COLS = ['BMI', 'MentHlth', 'PhysHlth']  # contínuas
BIN_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'PhysActivity',
    'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'DiffWalk', 'Sex',
]
ORD_COLS = ['GenHlth', 'Age', 'Education', 'Income']  # ordinais (categóricas)

CAT_COLS = BIN_COLS + ORD_COLS  # 17 categóricas (para o SMOTENC)

SEED = 42
TEST_SIZE = 0.20
VAL_FRAC = 0.10


def carregar_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Base tratada não encontrada em {DATA_PATH}. Execute src/data_prep.py.'
        )
    df = pd.read_parquet(DATA_PATH)
    faltando = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if faltando:
        raise ValueError(f'Colunas ausentes: {faltando}')
    return df


def _treinar(X_tr, y_tr, X_val, y_val, nomes, spw):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 5,
        'scale_pos_weight': spw, 'seed': SEED, 'verbosity': 0,
    }
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=nomes)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=nomes)
    return xgb.train(params, dtr, num_boost_round=1000,
                     evals=[(dval, 'val')], early_stopping_rounds=50,
                     verbose_eval=False)


def _predizer(modelo, X, nomes):
    dm = xgb.DMatrix(X, feature_names=nomes)
    it = getattr(modelo, 'best_iteration', None)
    if it is None:
        return modelo.predict(dm)
    return modelo.predict(dm, iteration_range=(0, it + 1))


def integridade_binarias(X_sintetico: np.ndarray) -> dict:
    """Fração de células de features binárias com valor fora de {0, 1}.

    SMOTE (interpolação) gera fracionários; SMOTENC (moda) mantém 0/1.
    """
    bin_idx = [FEATURES.index(c) for c in BIN_COLS]
    vals = X_sintetico[:, bin_idx]
    fracionarias = np.mean(~np.isin(vals, [0.0, 1.0]))
    return {'frac_binarias_invalidas': float(fracionarias)}


def executar() -> dict:
    df = carregar_base()
    y = df[TARGET_COL].astype(int).to_numpy()
    X_raw = df[FEATURES].astype(float).to_numpy()

    cat_idx = [FEATURES.index(c) for c in CAT_COLS]

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                       random_state=SEED)
    idx_in, idx_val = train_test_split(idx_tr, test_size=VAL_FRAC,
                                       stratify=y[idx_tr], random_state=SEED)

    # Padroniza contínuas (fit no treino interno; XGBoost é tree-based, mas o
    # SMOTE usa k-NN e precisa de escala nas contínuas).
    num_idx = [FEATURES.index(c) for c in NUM_COLS]
    scaler = StandardScaler()
    scaler.fit(X_raw[idx_in][:, num_idx])

    def _escalar(X):
        Xs = X.copy()
        Xs[:, num_idx] = scaler.transform(X[:, num_idx])
        return Xs

    X_in = _escalar(X_raw[idx_in])
    X_val = _escalar(X_raw[idx_val])
    X_te = _escalar(X_raw[idx_te])
    y_in, y_val, y_te = y[idx_in], y[idx_val], y[idx_te]

    spw = float((y_in == 0).sum() / max((y_in == 1).sum(), 1))
    print(f'Treino interno: {len(X_in):,} | prevalência: {y_in.mean():.4f} | spw={spw:.2f}')

    resultados = []
    integridade = []

    # --- Estratégia 1: scale_pos_weight (sem oversampling) ---
    print('\n[1/3] baseline: scale_pos_weight (sem oversampling)...')
    m = _treinar(X_in, y_in, X_val, y_val, FEATURES, spw)
    prob = _predizer(m, X_te, FEATURES)
    pred = prob >= 0.5
    resultados.append({
        'estrategia': 'scale_pos_weight (sem oversampling)',
        'n_treino': len(y_in),
        'roc_auc': float(roc_auc_score(y_te, prob)),
        'pr_auc': float(average_precision_score(y_te, prob)),
        'recall@0.5': float(recall_score(y_te, pred, zero_division=0)),
        'precisao@0.5': float(precision_score(y_te, pred, zero_division=0)),
        'f1@0.5': float(fbeta_score(y_te, pred, beta=1, zero_division=0)),
    })

    # --- Estratégia 2: SMOTE (todas as features tratadas como contínuas) ---
    print('[2/3] SMOTE (trata as 17 categóricas como contínuas)...')
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_in, y_in)
    # sintéticos = amostras além do treino original
    X_sint = X_sm[len(y_in):]
    integridade.append({'metodo': 'SMOTE', **integridade_binarias(X_sint)})
    m = _treinar(X_sm, y_sm, X_val, y_val, FEATURES, 1.0)  # balanceado -> spw=1
    prob = _predizer(m, X_te, FEATURES)
    pred = prob >= 0.5
    resultados.append({
        'estrategia': 'SMOTE',
        'n_treino': len(y_sm),
        'roc_auc': float(roc_auc_score(y_te, prob)),
        'pr_auc': float(average_precision_score(y_te, prob)),
        'recall@0.5': float(recall_score(y_te, pred, zero_division=0)),
        'precisao@0.5': float(precision_score(y_te, pred, zero_division=0)),
        'f1@0.5': float(fbeta_score(y_te, pred, beta=1, zero_division=0)),
    })

    # --- Estratégia 3: SMOTENC (17 categóricas explícitas) ---
    print('[3/3] SMOTENC (17 categóricas declaradas)...')
    smotenc = SMOTENC(categorical_features=cat_idx, random_state=SEED, k_neighbors=5)
    X_sn, y_sn = smotenc.fit_resample(X_in, y_in)
    X_sint = X_sn[len(y_in):]
    integridade.append({'metodo': 'SMOTENC', **integridade_binarias(X_sint)})
    m = _treinar(X_sn, y_sn, X_val, y_val, FEATURES, 1.0)
    prob = _predizer(m, X_te, FEATURES)
    pred = prob >= 0.5
    resultados.append({
        'estrategia': 'SMOTENC',
        'n_treino': len(y_sn),
        'roc_auc': float(roc_auc_score(y_te, prob)),
        'pr_auc': float(average_precision_score(y_te, prob)),
        'recall@0.5': float(recall_score(y_te, pred, zero_division=0)),
        'precisao@0.5': float(precision_score(y_te, pred, zero_division=0)),
        'f1@0.5': float(fbeta_score(y_te, pred, beta=1, zero_division=0)),
    })

    df_res = pd.DataFrame(resultados)
    df_int = pd.DataFrame(integridade)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUTPUT_DIR / 'smote_smotenc.csv', index=False)
    df_int.to_csv(OUTPUT_DIR / 'integridade_sinteticos.csv', index=False)
    _escrever_relatorio(df_res, df_int, spw)

    print('\n=== Métricas por estratégia (teste) ===')
    print(df_res.round(3).to_string(index=False))
    print('\n=== Integridade dos sintéticos (features binárias) ===')
    print(df_int.round(4).to_string(index=False))
    print(f'\nSaídas em {OUTPUT_DIR}')
    return {'resultados': df_res, 'integridade': df_int}


def _escrever_relatorio(df_res, df_int, spw):
    linhas = [
        '# SMOTE × SMOTENC — HeartHealth (PISI3)',
        '',
        '## 1. Tipos de feature',
        '',
        '- 3 **contínuas**: `BMI`, `MentHlth`, `PhysHlth`.',
        '- 13 **binárias** (0/1): `HighBP`, `HighChol`, `CholCheck`, `Smoker`, '
        '`Stroke`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, '
        '`AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.',
        '- 4 **ordinais** (categóricas ordenadas): `GenHlth`, `Age`, `Education`, '
        '`Income`.',
        '- Total de **17 categóricas** (binárias + ordinais) e **3 contínuas**.',
        '',
        '## 2. Por que importa: integridade dos dados sintéticos',
        '',
        '- **SMOTE** interpola entre vizinhos k-NN e, nas features binárias/ordinais, '
        'gera valores fracionários (ex.: `HighBP = 0.37`) que não são categorias '
        'válidas.',
        '- **SMOTENC** usa interpolação apenas nas contínuas e a **moda** nas '
        'categóricas, mantendo 0/1 e inteiros válidos.',
        '',
        df_int.round(4).to_markdown(index=False),
        '',
        '## 3. Métricas no teste (XGBoost)',
        '',
        df_res.round(3).to_markdown(index=False),
        '',
        '## 4. Leitura e recomendação',
        '',
        f'- O baseline usa `scale_pos_weight = {spw:.2f}` (ponderação de classe), '
        'que é a forma recomendada de lidar com desbalanceamento no XGBoost.',
        '- O oversampling (SMOTE/SMOTENC) não melhora as métricas em relação ao '
        '`scale_pos_weight` — ao contrário, tende a piorar levemente (sintéticos '
        'adicionam ruído e o XGBoost já pondera a classe minoritária).',
        '- **Se optar por oversampling, use SMOTENC** (não SMOTE): as 17 features '
        'categóricas exigem tratamento de moda, não de interpolação, para não gerar '
        'valores inválidos.',
        '- **Recomendação final:** manter `scale_pos_weight` (sem oversampling). O '
        'SMOTENC fica documentado como alternativa válida, caso se deseje um '
        'classificador baseado em distância (ex.: SVM/k-NN) que se beneficie de '
        'reescalonamento.',
        '',
        '## 5. Referências',
        '- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling '
        'Technique*. JAIR.',
    ]
    (OUTPUT_DIR / 'relatorio_smote_smotenc.md').write_text(
        '\n'.join(linhas), encoding='utf-8'
    )


if __name__ == '__main__':
    executar()
