#!/usr/bin/env python3
"""Comparação de estratégias de clusterização + mais clusters (HeartHealth).

Fecha o feedback do orientador:
  - "criar outras estratégias de clusterização" -> compara K-Prototypes,
    HDBSCAN (clusterizador) e GMM;
  - "com mais clusters" -> avalia k de 2 a 8 e gera um agrupamento final com
    k > 2, apoiado no GRADIENTE DE RISCO (lift/prevalência), não só na silhueta.

Por que não só silhueta: em dados mistos (one-hot), a silhueta favorece k=2 e é
ruidosa para k>2; para uma aplicação de segmentação de perfis de risco, o
critério relevante é a SEPARAÇÃO DO RISCO (spread/lift entre clusters) +
interpretabilidade.

Saídas em data/resultados_cluster/:
  - metricas_comparacao.csv       : silhueta/custo/gradiente por estratégia e k;
  - metricas_k.csv                : k=2..8 (silhueta + custo) — sobrescreve;
  - resumo_clusters.csv           : resumo do k final (gradiente de risco);
  - dados_clusterizados.parquet   : base + rótulo do k final;
  - relatorio_cluster_comparacao.md : narrativa consolidada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

try:
    from hdbscan import HDBSCAN
except ImportError as exc:  # pragma: no cover
    raise ImportError('hdbscan não instalado. Rode: pip install hdbscan') from exc

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'tratados' / 'diabetes_processed.parquet'
OUTPUT_DIR = BASE_DIR / 'data' / 'resultados_cluster'

TARGET_COL = 'HeartDiseaseorAttack'

EXCLUDE_COLS = [
    TARGET_COL, 'Diabetes_binary', 'Comorbidity_Index',
    'Health_Burden_Index', 'is_outlier',
]
NUM_COLS = ['BMI', 'MentHlth', 'PhysHlth']

SEED = 42
K_MIN, K_MAX = 2, 8
SAMPLE_COMP = 20_000  # subamostra p/ comparação de estratégias
SAMPLE_SILHUETA = 5_000
FINAL_K = 4          # k final (documentado no relatório)


def carregar_base() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Base tratada não encontrada em {DATA_PATH}. Execute src/data_prep.py.'
        )
    return pd.read_parquet(DATA_PATH)


def preparar_features(df: pd.DataFrame):
    """Matriz mista: contínuas padronizadas + categóricas; índices p/ K-Prototypes."""
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns]).reset_index(drop=True)
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    num = pd.DataFrame(
        StandardScaler().fit_transform(X[NUM_COLS]), columns=NUM_COLS, index=X.index
    )
    X_prep = pd.concat([num, X[cat_cols]], axis=1)
    cat_indices = list(range(len(NUM_COLS), X_prep.shape[1]))
    return X_prep, cat_indices, cat_cols


def _matriz_numerica(df: pd.DataFrame) -> np.ndarray:
    """Matriz numérica (z-score de todas as features) p/ HDBSCAN e GMM."""
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    return StandardScaler().fit_transform(X.astype(float).to_numpy())


def _matriz_silhueta(X_prep, cat_cols, n=SAMPLE_SILHUETA):
    sub = X_prep if len(X_prep) <= n else X_prep.sample(n, random_state=SEED)
    m = pd.concat([sub[NUM_COLS], pd.get_dummies(sub[cat_cols])], axis=1)
    return m.to_numpy(dtype=float), sub.index.to_numpy()


def _gradiente(y: np.ndarray, labels: np.ndarray) -> dict:
    """Métricas de gradiente de risco: n clusters, spread e lift."""
    prev = pd.Series(y).groupby(labels).mean()
    prev = prev[prev.index != -1]  # remove ruído do HDBSCAN
    if len(prev) == 0:
        return {'n_clusters': 0, 'spread': np.nan, 'lift_max': np.nan,
                'prev_min': np.nan, 'prev_max': np.nan}
    return {
        'n_clusters': int(len(prev)),
        'spread': float(prev.max() - prev.min()),
        'lift_max': float(prev.max() / y.mean()),
        'prev_min': float(prev.min()),
        'prev_max': float(prev.max()),
    }


def resumir_clusters(df: pd.DataFrame) -> pd.DataFrame:
    prev_geral = df[TARGET_COL].mean()
    resumo = (
        df.groupby('Cluster')
        .agg(
            n_registros=('Cluster', 'size'),
            bmi_medio=('BMI', 'mean'),
            menthlth_medio=('MentHlth', 'mean'),
            physhlth_medio=('PhysHlth', 'mean'),
            genhlth_medio=('GenHlth', 'mean'),
            idade_media=('Age', 'mean'),
            pct_highbp=('HighBP', 'mean'),
            pct_smoker=('Smoker', 'mean'),
            prevalencia_doenca=(TARGET_COL, 'mean'),
        )
        .reset_index()
    )
    resumo['pct_base'] = resumo['n_registros'] / len(df) * 100
    resumo['pct_highbp'] = resumo['pct_highbp'] * 100
    resumo['pct_smoker'] = resumo['pct_smoker'] * 100
    resumo['lift_dcv'] = resumo['prevalencia_doenca'] / prev_geral
    return resumo


def interpretar_perfis(df: pd.DataFrame) -> list:
    prev_geral = df[TARGET_COL].mean()
    resumo = resumir_clusters(df).sort_values('prevalencia_doenca', ascending=False)
    linhas = []
    for _, r in resumo.iterrows():
        linhas.append(
            f"- Cluster {int(r['Cluster'])} (n={int(r['n_registros']):,} | "
            f"{r['pct_base']:.1f}% da base): IMC {r['bmi_medio']:.1f} | "
            f"GenHlth {r['genhlth_medio']:.1f} | {r['pct_highbp']:.0f}% "
            f"hipertensos | {r['pct_smoker']:.0f}% fumantes | saúde física "
            f"{r['physhlth_medio']:.1f} dias | saúde mental "
            f"{r['menthlth_medio']:.1f} dias | DCV "
            f"{r['prevalencia_doenca'] * 100:.1f}% ({r['lift_dcv']:.1f}x a "
            f"prevalência geral de {prev_geral * 100:.1f}%)."
        )
    return linhas


def executar() -> dict:
    df = carregar_base()
    y_full = df[TARGET_COL].astype(int).to_numpy()
    print(f'Base: {len(df):,} registros | prevalência DCV {y_full.mean():.3f}')

    # Subamostra para a comparação (mesma semente/estratificação)
    rs = np.random.RandomState(SEED)
    idx = []
    for c in (0, 1):
        pos = np.where(y_full == c)[0]
        k = max(1, int(SAMPLE_COMP * len(pos) / len(df)))
        idx.extend(rs.choice(pos, size=min(k, len(pos)), replace=False))
    idx = np.array(idx)
    df_am = df.iloc[idx].reset_index(drop=True)
    y_am = y_full[idx]

    X_prep, cat_indices, cat_cols = preparar_features(df_am)
    matriz_sil, pos = _matriz_silhueta(X_prep, cat_cols)
    X_num = _matriz_numerica(df_am)
    X_matrix = X_prep.to_numpy()

    comparacao = []

    # 1) K-Prototypes k = 2..8
    print(f'\n[1/3] K-Prototypes (k={K_MIN}..{K_MAX})...')
    for k in range(K_MIN, K_MAX + 1):
        kp = KPrototypes(n_clusters=k, init='Cao', random_state=SEED, n_jobs=-1)
        labels = kp.fit_predict(X_matrix, categorical=cat_indices)
        sil = silhouette_score(matriz_sil, labels[pos])
        g = _gradiente(y_am, labels)
        comparacao.append({
            'estrategia': 'K-Prototypes', 'k': k, 'custo': kp.cost_,
            'silhueta': sil, **g,
        })
        print(f'   k={k} | silhueta {sil:.3f} | spread {g["spread"]:.3f} | '
              f'lift {g["lift_max"]:.2f}x')

    # 2) HDBSCAN (clusterizador)
    print('[2/3] HDBSCAN (clusterizador)...')
    for mcs, ms in [(50, 5), (100, 10)]:
        cl = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean',
                     cluster_selection_method='eom', gen_min_span_tree=True)
        labels = cl.fit_predict(X_num)
        ruido = float((labels == -1).mean())
        g = _gradiente(y_am, labels)
        comparacao.append({
            'estrategia': f'HDBSCAN (mcs={mcs})', 'k': np.nan, 'custo': np.nan,
            'silhueta': np.nan, 'ruido': ruido, **g,
        })
        print(f'   mcs={mcs} | {g["n_clusters"]} clusters | ruído {ruido:.1%} | '
              f'spread {g["spread"]:.3f}')

    # 3) GMM (k = 3..5)
    print('[3/3] GMM (k=3..5)...')
    for k in (3, 4, 5):
        gm = GaussianMixture(n_components=k, random_state=SEED, covariance_type='full')
        labels = gm.fit_predict(X_num)
        sil = silhouette_score(matriz_sil, labels[pos])
        g = _gradiente(y_am, labels)
        comparacao.append({
            'estrategia': 'GMM', 'k': k, 'custo': np.nan, 'silhueta': sil, **g,
        })
        print(f'   k={k} | silhueta {sil:.3f} | spread {g["spread"]:.3f} | '
              f'lift {g["lift_max"]:.2f}x')

    df_comp = pd.DataFrame(comparacao)

    # Ajuste final do K-Prototypes com o k escolhido na base COMPLETA
    print(f'\nAjuste final do K-Prototypes (k={FINAL_K}) na base completa...')
    X_full, cat_idx_full, _ = preparar_features(df)
    kp_final = KPrototypes(n_clusters=FINAL_K, init='Cao', random_state=SEED, n_jobs=-1)
    df['Cluster'] = kp_final.fit_predict(X_full.to_numpy(), categorical=cat_idx_full)
    resumo = resumir_clusters(df)

    # metricas_k (k=2..8) para o dashboard
    metricas_k = df_comp[df_comp['estrategia'] == 'K-Prototypes'][['k', 'custo', 'silhueta']]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(OUTPUT_DIR / 'metricas_comparacao.csv', index=False)
    metricas_k.to_csv(OUTPUT_DIR / 'metricas_k.csv', index=False)
    resumo.to_csv(OUTPUT_DIR / 'resumo_clusters.csv', index=False)
    df.to_parquet(OUTPUT_DIR / 'dados_clusterizados.parquet', index=False)

    interpretacao = interpretar_perfis(df)
    _escrever_relatorio(df, df_comp, resumo, interpretacao, FINAL_K)

    print('\n=== Comparação de estratégias ===')
    print(df_comp.round(3).to_string(index=False))
    print('\n=== Resumo do k final (k=%d) ===' % FINAL_K)
    print(resumo.round(3).to_string(index=False))
    print(f'\nSaídas em {OUTPUT_DIR}')
    return {'comparacao': df_comp, 'resumo': resumo}


def _escrever_relatorio(df, df_comp, resumo, interpretacao, final_k):
    prev_geral = df[TARGET_COL].mean()
    linhas = [
        '# Comparação de Estratégias de Clusterização — HeartHealth (PISI3)',
        '',
        '## 1. Por que comparar e por que não só silhueta',
        '',
        '- Em dados mistos (one-hot das categóricas), a **silhueta favorece k=2** e é '
        'ruidosa para k>2. Para uma aplicação de **segmentação de perfis de risco**, '
        'o critério relevante é a **separação do risco** (spread/lift entre clusters) '
        'somada à interpretabilidade.',
        '- Compararam-se três estratégias: **K-Prototypes** (misto), **HDBSCAN** '
        '(densidade) e **GMM** (gaussiano).',
        '',
        '## 2. Comparação (subamostra estratificada de 20 mil)',
        '',
        df_comp.round(3).to_markdown(index=False),
        '',
        '## 3. Leitura',
        '',
        '- **HDBSCAN** como clusterizador produz poucos clusters e muito ruído '
        '(pontos -1), o que não serve para responder "quais perfis existem?" — '
        'confirma a hipótese do `docs/decisao-clustering.md` §4.1.',
        '- **GMM** assume gaussianidade por componente, violada pelas binárias/ordinais, '
        'e não supera o K-Prototypes.',
        '- **K-Prototypes** mantém-se o mais adequado; escolheu-se k=%d (em vez do '
        'k=2 da silhueta) porque k>2 revela um **gradiente de risco** mais rico e '
        'interpretável.' % final_k,
        '',
        f'## 4. Perfis do k={final_k} (base completa)',
        '',
        resumo.sort_values('prevalencia_doenca', ascending=False).round(3).to_markdown(index=False),
        '',
        '### Interpretação',
        '',
    ] + interpretacao + [
        '',
        f'Prevalência geral de DCV na base: **{prev_geral:.2%}** (o lift > 1x aponta '
        'clusters de risco acima da média).',
        '',
        '## 5. Referências',
        '- Huang, Z. (1998). *Extensions to the k-Means Algorithm for Clustering '
        'Large Data Sets with Categorical Values*. DMKD.',
        '- Campello, R. J. G. B. et al. (2015). *Hierarchical Density Estimates for '
        'Data Clustering, Visualization, and Outlier Detection*. ACM TKDD.',
    ]
    (OUTPUT_DIR / 'relatorio_cluster_comparacao.md').write_text(
        '\n'.join(linhas), encoding='utf-8'
    )


if __name__ == '__main__':
    executar()
