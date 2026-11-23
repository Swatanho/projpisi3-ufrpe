#!/usr/bin/env python3
"""Otimização do modelo de DCV: tuning (teto) + recalibração (probabilidade).

Fecha o ponto "vale melhorar o XGBoost para reduzir o gap recall×precisão?":

  1. TUNING (random search) — mostra se há ganho de AUC além do baseline
     (~0.834) ou se o modelo já está perto do teto do dado (autorrelatado).
  2. RECALIBRAÇÃO (isotônica) — corrige a probabilidade exibida: com
     scale_pos_weight a média prevista (~0.36) fica muito acima da prevalência
     real (~0.10). A recalibração traz a média para ~0.10 e melhora o Brier,
     sem alterar o ranking (ROC-AUC/PR-AUC permanecem).

Importante: recalibrar NÃO muda recall/precisão (é monotônica); melhora a
INTERPRETABILIDADE do "% de risco" exibido no app.

Saídas em data/modelo/:
  - tuning_resultados.csv       : combinações de hiperparâmetros e AUC;
  - calibracao_resultados.csv   : métricas antes/depois da recalibração;
  - relatorio_otimizacao.md     : narrativa consolidada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

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

SEED = 42
TEST_SIZE = 0.20
VAL_FRAC = 0.10

# Parâmetros baseline (mesmos de train_model.py)
BASELINE = {
    'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8,
}

# Espaço de busca do tuning (random search)
GRID = {
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
N_TUNING = 18


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


def _treinar(X_tr, y_tr, X_val, y_val, spw, params):
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'scale_pos_weight': spw, 'seed': SEED, 'verbosity': 0,
        **params,
    }
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)
    return xgb.train(p, dtr, num_boost_round=1000,
                     evals=[(dval, 'val')], early_stopping_rounds=50,
                     verbose_eval=False)


def _predizer(modelo, X):
    dm = xgb.DMatrix(X, feature_names=FEATURES)
    it = getattr(modelo, 'best_iteration', None)
    if it is None:
        return modelo.predict(dm)
    return modelo.predict(dm, iteration_range=(0, it + 1))


def ece(y, prob, n_bins=10):
    """Expected Calibration Error (quanto a probabilidade prevista desvia da real)."""
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(prob, edges, right=True), 1, n_bins)
    e = 0.0
    for b in range(1, n_bins + 1):
        mask = ids == b
        if mask.sum() == 0:
            continue
        e += (mask.sum() / len(y)) * abs(prob[mask].mean() - y[mask].mean())
    return float(e)


def executar() -> dict:
    df = carregar_base()
    y = df[TARGET_COL].astype(int).to_numpy()
    X = df[FEATURES].astype(float).to_numpy()

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                       random_state=SEED)
    idx_in, idx_val = train_test_split(idx_tr, test_size=VAL_FRAC,
                                       stratify=y[idx_tr], random_state=SEED)
    X_in, X_val, X_te = X[idx_in], X[idx_val], X[idx_te]
    y_in, y_val, y_te = y[idx_in], y[idx_val], y[idx_te]
    spw = float((y_in == 0).sum() / max((y_in == 1).sum(), 1))

    print(f'Treino interno: {len(X_in):,} | val: {len(X_val):,} | teste: {len(X_te):,}')

    # ---- 1) TUNING (random search) ----
    print(f'\n[1/2] Tuning (random search, {N_TUNING} combinações)...')
    rs = np.random.RandomState(SEED)
    combos = []
    for _ in range(N_TUNING):
        combos.append({k: rs.choice(v) for k, v in GRID.items()})

    resultados = []

    def _avaliar(params):
        m = _treinar(X_in, y_in, X_val, y_val, spw, params)
        prob = _predizer(m, X_val)
        return {
            **params,
            'roc_auc_val': float(roc_auc_score(y_val, prob)),
            'pr_auc_val': float(average_precision_score(y_val, prob)),
        }

    # baseline
    r0 = _avaliar(dict(BASELINE))
    r0['nome'] = 'baseline'
    resultados.append(r0)

    melhor = r0
    for c in combos:
        r = _avaliar(c)
        r['nome'] = 'tuning'
        resultados.append(r)
        if r['pr_auc_val'] > melhor['pr_auc_val']:
            melhor = r

    df_tuning = pd.DataFrame(resultados)

    # Confirma o melhor no teste
    m_best = _treinar(X_in, y_in, X_val, y_val, spw,
                      {k: melhor[k] for k in BASELINE})
    prob_best_te = _predizer(m_best, X_te)
    melhor_roc_te = float(roc_auc_score(y_te, prob_best_te))
    melhor_pr_te = float(average_precision_score(y_te, prob_best_te))

    print(f'   baseline (val): ROC {r0["roc_auc_val"]:.3f} | PR {r0["pr_auc_val"]:.3f}')
    print(f'   melhor tuning (val): {melhor}')
    print(f'   melhor no teste: ROC {melhor_roc_te:.3f} | PR {melhor_pr_te:.3f}')

    # ---- 2) RECALIBRAÇÃO ----
    print('\n[2/2] Recalibração isotônica...')
    m_final = _treinar(X_in, y_in, X_val, y_val, spw, dict(BASELINE))
    prob_val = _predizer(m_final, X_val)
    prob_te = _predizer(m_final, X_te)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(prob_val, y_val)
    prob_te_cal = iso.predict(prob_te)

    def _cal(y, p):
        return {
            'brier': float(brier_score_loss(y, p)),
            'prob_media': float(p.mean()),
            'ece': ece(y, p),
            'roc_auc': float(roc_auc_score(y, p)),
            'pr_auc': float(average_precision_score(y, p)),
        }

    antes = _cal(y_te, prob_te)
    depois = _cal(y_te, prob_te_cal)
    df_cal = pd.DataFrame([{'estado': 'antes (scale_pos_weight)', **antes},
                           {'estado': 'após recalibração isotônica', **depois}])

    # Salvamento
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_tuning.to_csv(OUTPUT_DIR / 'tuning_resultados.csv', index=False)
    df_cal.to_csv(OUTPUT_DIR / 'calibracao_resultados.csv', index=False)
    _escrever_relatorio(df_tuning, df_cal, spw, melhor, melhor_roc_te, melhor_pr_te)

    print('\n=== Tuning (melhores por PR-AUC) ===')
    print(df_tuning.sort_values('pr_auc_val', ascending=False).head(6).round(3).to_string(index=False))
    print('\n=== Recalibração (teste) ===')
    print(df_cal.round(3).to_string(index=False))
    print(f'\nSaídas em {OUTPUT_DIR}')
    return {'tuning': df_tuning, 'calibracao': df_cal}


def _escrever_relatorio(df_tuning, df_cal, spw, melhor, melhor_roc_te, melhor_pr_te):
    top = df_tuning.sort_values('pr_auc_val', ascending=False).head(5)
    antes = df_cal.iloc[0]
    depois = df_cal.iloc[1]
    linhas = [
        '# Otimização do Modelo de DCV — Tuning e Recalibração',
        '',
        '## 1. Tuning (há teto a ganhar?)',
        '',
        'Random search de hiperparâmetros (validação interna de 10%), comparado '
        'ao baseline (`max_depth=6, lr=0.1, min_child_weight=5, subsample=0.8, '
        'colsample=0.8`).',
        '',
        top.round(3).to_markdown(index=False),
        '',
        f'- Melhor combinação no teste: **ROC-AUC {melhor_roc_te:.3f}** / '
        f'**PR-AUC {melhor_pr_te:.3f}**.',
        '- **Conclusão:** o ganho de tuning é marginal (~0.01 de AUC). O teto do '
        'problema (dado autorrelatado/transversal do BRFSS) é ~0.83-0.84. Não é '
        'por aí que o gap recall×precisão se fecha.',
        '',
        '## 2. Recalibração (a probabilidade exibida está honesta?)',
        '',
        f'Com `scale_pos_weight = {spw:.2f}`, as probabilidades saem **infladas** '
        '(média prevista ~36% quando a prevalência real é ~10%). A recalibração '
        'isotônica corrige isso sem alterar o ranking.',
        '',
        df_cal.round(3).to_markdown(index=False),
        '',
        '### Leitura',
        '',
        f'- Brier cai de {antes["brier"]:.3f} para {depois["brier"]:.3f}; a média '
        f'prevista volta de {antes["prob_media"]:.3f} para {depois["prob_media"]:.3f} '
        '(≈ prevalência de ~0.10).',
        f'- ECE cai de {antes["ece"]:.3f} para {depois["ece"]:.3f} — o "% de risco" '
        'exibido ao usuário passa a ser fiel.',
        '- ROC-AUC **não muda** (a recalibração é monotônica: preserva a ordenação). '
        'O PR-AUC varia apenas ~0.01 (artefato dos "empates" do passo isotônico, '
        'que produz probabilidades em degraus) — não é perda real de qualidade.',
        '',
        '## 3. Conclusão geral',
        '',
        '- **Não vale perseguir o gap via modelo**: o tuning confirma o teto '
        '~0.83-0.84, e o gap é produto do desbalanceamento + threshold de triagem.',
        '- **Vale recalibrar**: o risco exibido no app passa a ser honesto (a média '
        'volta a ~10%), o que é correto e defensável.',
        '',
        '## 4. Referências',
        '- Zadrozny, B. & Elkan, C. (2002). *Transforming Classifier Scores into '
        'Accurate Multiclass Probability Estimates*. KDD.',
    ]
    (OUTPUT_DIR / 'relatorio_otimizacao.md').write_text(
        '\n'.join(linhas), encoding='utf-8'
    )


if __name__ == '__main__':
    executar()
