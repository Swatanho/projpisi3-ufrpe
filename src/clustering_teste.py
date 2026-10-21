"""Segmentação de perfis com K-Prototypes sobre a base tratada (HeartHealth).

Executa o K-Prototypes usando as 20 variáveis do contrato do app (exceto
o alvo HeartDiseaseorAttack), com k escolhido pela silhueta (apoiado no
método do cotovelo) e interpretação dos perfis gerada dos resultados.

Saídas em data/resultados_cluster/:
- dados_clusterizados.parquet : base completa + rótulo de cluster;
- resumo_clusters.csv         : métricas por cluster (perfil e prevalência);
- metricas_k.csv              : custo e silhueta por k (escolha do k);
- relatorio_cluster_kprototypes.txt : relatório com interpretação gerada.
"""

from pathlib import Path

import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tratados" / "diabetes_processed.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "resultados_cluster"

TARGET_COL = "HeartDiseaseorAttack"

# Excluídas do clustering: o alvo (leakage), Diabetes_binary (fora do
# contrato de 20 features do app), colunas derivadas e flag de outlier.
EXCLUDE_COLS = [
    TARGET_COL,
    "Diabetes_binary",
    "Comorbidity_Index",
    "Health_Burden_Index",
    "is_outlier",
]

# Contínuas -> distância euclidiana (padronizadas); demais -> matching (categóricas)
NUM_COLS = ["BMI", "MentHlth", "PhysHlth"]

K_MIN, K_MAX = 2, 6  # faixa de k avaliada
SAMPLE_K = 10_000  # amostra p/ escolha do k (cotovelo + silhueta)
SAMPLE_SILHUETA = 5_000  # subamostra p/ cálculo da silhueta
SEED = 42


def load_treated_data() -> pd.DataFrame:
    """Carrega a base tratada pela pipeline (src/data_prep.py)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo tratado não encontrado em: {DATA_PATH}. "
            "Execute primeiro o script 'src/data_prep.py'."
        )
    return pd.read_parquet(DATA_PATH)


def preparar_features(df: pd.DataFrame) -> tuple:
    """Monta a matriz de features: contínuas padronizadas + categóricas/binárias.

    Retorna (X_prep, cat_indices, cat_cols) com as numéricas nas primeiras
    posições e as categóricas nas demais (índices exigidos pelo K-Prototypes).
    """
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.reset_index(drop=True)
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    num_parte = pd.DataFrame(
        StandardScaler().fit_transform(X[NUM_COLS]),
        columns=NUM_COLS,
        index=X.index,
    )
    X_prep = pd.concat([num_parte, X[cat_cols]], axis=1)
    cat_indices = list(range(len(NUM_COLS), X_prep.shape[1]))
    return X_prep, cat_indices, cat_cols


def _matriz_silhueta(X_prep: pd.DataFrame, cat_cols: list) -> tuple:
    """Amostra fixa e matriz (contínuas + one-hot) p/ silhueta euclidiana.

    O one-hot + distância euclidiana aproxima o matching (Hamming) usado
    pelo K-Prototypes nas variáveis categóricas.
    """
    sub = X_prep
    if len(X_prep) > SAMPLE_SILHUETA:
        sub = X_prep.sample(SAMPLE_SILHUETA, random_state=SEED)
    matriz = pd.concat([sub[NUM_COLS], pd.get_dummies(sub[cat_cols])], axis=1)
    return matriz.to_numpy(dtype=float), sub.index.to_numpy()


def calcular_metricas_k(X_prep, cat_indices, cat_cols, k_min=K_MIN, k_max=K_MAX):
    """Roda o K-Prototypes por k e devolve (k_escolhido, df_metricas).

    O k final é o de maior silhueta; a curva de custo (cotovelo) fica
    registrada em df_metricas como apoio à decisão.
    """
    matriz_sil, pos = _matriz_silhueta(X_prep, cat_cols)
    X_matrix = X_prep.to_numpy()
    registros = []
    print(f"   k de {k_min} a {k_max} (amostra de {len(X_prep):,} registros):")
    for k in range(k_min, k_max + 1):
        kp = KPrototypes(n_clusters=k, init="Cao", random_state=SEED, n_jobs=-1)
        labels = kp.fit_predict(X_matrix, categorical=cat_indices)
        sil = silhouette_score(matriz_sil, labels[pos])
        registros.append({"k": k, "custo": kp.cost_, "silhueta": sil})
        print(f"   - k={k} | custo: {kp.cost_:.1f} | silhueta: {sil:.3f}")
    df_metricas = pd.DataFrame(registros)
    k_escolhido = int(df_metricas.loc[df_metricas["silhueta"].idxmax(), "k"])
    return k_escolhido, df_metricas


def resumir_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por cluster usadas no relatório e no dashboard."""
    prev_geral = df[TARGET_COL].mean()
    resumo = (
        df.groupby("Cluster")
        .agg(
            n_registros=("Cluster", "size"),
            bmi_medio=("BMI", "mean"),
            menthlth_medio=("MentHlth", "mean"),
            physhlth_medio=("PhysHlth", "mean"),
            genhlth_medio=("GenHlth", "mean"),
            idade_media=("Age", "mean"),
            pct_highbp=("HighBP", "mean"),
            pct_smoker=("Smoker", "mean"),
            prevalencia_doenca=(TARGET_COL, "mean"),
        )
        .reset_index()
    )
    resumo["pct_base"] = resumo["n_registros"] / len(df) * 100
    resumo["pct_highbp"] = resumo["pct_highbp"] * 100
    resumo["pct_smoker"] = resumo["pct_smoker"] * 100
    resumo["lift_dcv"] = resumo["prevalencia_doenca"] / prev_geral
    return resumo


def interpretar_perfis(df: pd.DataFrame) -> list:
    """Gera a leitura dos perfis a partir dos resultados reais (sem texto fixo)."""
    prev_geral = df[TARGET_COL].mean()
    resumo = resumir_clusters(df).sort_values("prevalencia_doenca", ascending=False)
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


def salvar_resultados(df, resumo, df_metricas, k_escolhido) -> Path:
    """Salva dados clusterizados, resumos e relatório em data/resultados_cluster/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR / "dados_clusterizados.parquet", index=False)
    resumo.to_csv(OUTPUT_DIR / "resumo_clusters.csv", index=False)
    df_metricas.to_csv(OUTPUT_DIR / "metricas_k.csv", index=False)

    interpretacao = interpretar_perfis(df)
    prev_geral = df[TARGET_COL].mean()
    with (OUTPUT_DIR / "relatorio_cluster_kprototypes.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("Relatório de Análise de Clusters - K-Prototypes\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {DATA_PATH.name} (base tratada - src/data_prep.py)\n")
        f.write(f"Registros analisados: {len(df):,}\n")
        f.write(f"Número de clusters: {k_escolhido} (maior silhueta; ver metricas_k.csv)\n")
        f.write(f"Prevalência geral de DCV: {prev_geral * 100:.2f}%\n")
        f.write("Features: 20 do contrato do app (excluídos o alvo, "
                "Diabetes_binary, colunas derivadas e is_outlier)\n\n")
        f.write("Resumo por Cluster\n")
        f.write(resumo.to_string(index=False) + "\n\n")
        f.write("Interpretação dos perfis (ordenada por prevalência de DCV)\n")
        for linha in interpretacao:
            f.write(linha + "\n")
        f.write(
            "\nObservação: K-Prototypes foi escolhido porque o conjunto "
            "possui variáveis numéricas e categóricas/binárias misturadas. "
            "KMeans, em sua forma padrão, exige variáveis numéricas e perde "
            "interpretabilidade em dados mistos. O alvo HeartDiseaseorAttack "
            "foi excluído das features para evitar leakage.\n"
        )
    return OUTPUT_DIR


def executar_clustering(sample_k=SAMPLE_K, seed=SEED):
    """Pipeline completa: escolha do k, ajuste final na base e relatórios."""
    print("1. Carregando a base tratada (parquet)...")
    df = load_treated_data()
    print(f"   Base tratada: {len(df):,} registros.")

    print("2. Escolha do k (cotovelo + silhueta)...")
    df_amostra = df.sample(min(sample_k, len(df)), random_state=seed)
    X_prep, cat_indices, cat_cols = preparar_features(df_amostra)
    k_escolhido, df_metricas = calcular_metricas_k(X_prep, cat_indices, cat_cols)

    print(f"3. Ajuste final do K-Prototypes (k={k_escolhido}) na base completa...")
    X_prep_full, cat_indices_full, _ = preparar_features(df)
    kp_final = KPrototypes(
        n_clusters=k_escolhido, init="Cao", random_state=seed, n_jobs=-1
    )
    df["Cluster"] = kp_final.fit_predict(
        X_prep_full.to_numpy(), categorical=cat_indices_full
    )

    resumo = resumir_clusters(df)
    print("\n=== Resumo por Cluster ===")
    print(resumo.to_string(index=False))

    print("\n4. Salvando resultados...")
    salvar_resultados(df, resumo, df_metricas, k_escolhido)
    print(f"   Saídas salvas em: {OUTPUT_DIR}")

    return df, resumo, df_metricas


if __name__ == "__main__":
    executar_clustering()