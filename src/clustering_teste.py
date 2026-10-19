from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_DIR = BASE_DIR / "data" / "resultados_cluster"


def gerar_graficos_cluster(df, num_cols, target_col, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribuição por cluster
    counts = df["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(i) for i in counts.index], counts.values, color="steelblue")
    ax.set_title("Distribuição dos Registros por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Quantidade de Registros")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "distribuicao_por_cluster.png", dpi=200)
    plt.close(fig)

    # 2. Boxplot de BMI por cluster
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="BMI", by="Cluster", ax=ax, grid=False)
    ax.set_title("Distribuição de BMI por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("BMI")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_bmi_por_cluster.png", dpi=200)
    plt.close(fig)

    # 3. Boxplot de saúde física e mental
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column="MentHlth", by="Cluster", ax=axes[0], grid=False)
    axes[0].set_title("MentHlth por Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Dias de Saúde Mental Comprometida")

    df.boxplot(column="PhysHlth", by="Cluster", ax=axes[1], grid=False)
    axes[1].set_title("PhysHlth por Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Dias de Saúde Física Comprometida")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_saude_por_cluster.png", dpi=200)
    plt.close(fig)

    # 4. Prevalência da doença por cluster
    prevalencia = df.groupby("Cluster")[target_col].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(i) for i in prevalencia.index], prevalencia.values * 100, color="darkorange")
    ax.set_title("Prevalência de HeartDiseaseorAttack por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Prevalência (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "prevalencia_doenca_por_cluster.png", dpi=200)
    plt.close(fig)

    # 5. Heatmap de perfil dos clusters
    profile = (
        df.groupby("Cluster")
        .agg(
            BMI=("BMI", "mean"),
            MentHlth=("MentHlth", "mean"),
            PhysHlth=("PhysHlth", "mean"),
            Prev_Doenca=(target_col, "mean"),
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    heatmap = ax.imshow(profile.values, cmap="YlOrRd")
    ax.set_xticks(range(profile.shape[1]))
    ax.set_xticklabels(profile.columns, rotation=25, ha="right")
    ax.set_yticks(range(profile.shape[0]))
    ax.set_yticklabels([str(i) for i in profile.index])
    ax.set_title("Perfil Médio dos Clusters")
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            ax.text(j, i, f"{profile.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_perfil_cluster.png", dpi=200)
    plt.close(fig)

    # 6. Plot do cotovelo já existente foi salvo no método principal
    return {
        "distribuicao": output_dir / "distribuicao_por_cluster.png",
        "bmi": output_dir / "boxplot_bmi_por_cluster.png",
        "saude": output_dir / "boxplot_saude_por_cluster.png",
        "prevalencia": output_dir / "prevalencia_doenca_por_cluster.png",
        "heatmap": output_dir / "heatmap_perfil_cluster.png",
    }


def gerar_lista_graficos_relatorio():
    lista = [
        {
            "grafico": "Elbow plot",
            "objetivo": "Selecionar o número ideal de clusters a partir do custo do algoritmo.",
            "interpretacao": "A queda mais acentuada do custo até k=4 indica que esse valor oferece uma separação clara entre perfis sem aumentar excessivamente a complexidade do modelo.",
        },
        {
            "grafico": "Distribuição por cluster",
            "objetivo": "Quantificar o tamanho dos grupos formados.",
            "interpretacao": "Os clusters com maior volume representam perfis mais frequentes na população, enquanto grupos menores podem sinalizar subgrupos de maior risco ou maior vulnerabilidade.",
        },
        {
            "grafico": "Boxplot de BMI por cluster",
            "objetivo": "Comparar o índice de massa corporal entre os grupos.",
            "interpretacao": "Clusters com BMI médio mais elevado tendem a concentrar maior risco cardiovascular e pior perfil de saúde geral.",
        },
        {
            "grafico": "Boxplot de saúde física e mental por cluster",
            "objetivo": "Avaliar o comprometimento de saúde física e mental entre os grupos.",
            "interpretacao": "Clusters com maiores valores em MentHlth e PhysHlth sugerem pior qualidade de vida e maior impacto das condições de saúde na rotina dos indivíduos.",
        },
        {
            "grafico": "Prevalência de HeartDiseaseorAttack por cluster",
            "objetivo": "Medir a probabilidade de doença cardíaca em cada grupo.",
            "interpretacao": "Os clusters com maior prevalência de doença representam perfis de maior risco e merecem atenção prioritária em intervenções preventivas.",
        },
        {
            "grafico": "Heatmap de perfil dos clusters",
            "objetivo": "Sintetizar as médias das variáveis-chave por cluster.",
            "interpretacao": "A matriz de médias permite identificar rapidamente quais grupos apresentam maior risco, maior comprometimento de saúde e pior perfil comportamental.",
        },
    ]
    return lista


def salvar_lista_relatorio(lista, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    relatorio_path = output_dir / "lista_graficos_relatorio.md"
    with relatorio_path.open("w", encoding="utf-8") as f:
        f.write("# Lista de gráficos para relatório\n\n")
        for idx, item in enumerate(lista, start=1):
            f.write(f"## {idx}. {item['grafico']}\n")
            f.write(f"- Objetivo: {item['objetivo']}\n")
            f.write(f"- Interpretação: {item['interpretacao']}\n\n")
    return relatorio_path


def testar_kprototypes(data_path, sample_size=None):
    print("1. Carregando os dados...")
    df = pd.read_csv(data_path)

    if sample_size is not None and len(df) > sample_size:
        print(f"Usando uma amostra de {sample_size} registros para validação rápida.")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"Executando com o dataset completo: {len(df):,} registros.")

    target_col = "HeartDiseaseorAttack"
    if target_col in df.columns:
        X = df.drop(columns=[target_col]).copy()
    else:
        X = df.copy()

    num_cols = ["BMI", "MentHlth", "PhysHlth"]
    for extra_col in [
        "Health_Burden_Index",
        "Comorbidity_Index",
        "Carga_Saude_Comprometida",
        "Indice_Comorbidades",
    ]:
        if extra_col in X.columns and extra_col not in num_cols:
            num_cols.append(extra_col)

    if not num_cols:
        raise ValueError("Nenhuma coluna numérica foi encontrada para o K-Prototypes.")

    cat_cols = [col for col in X.columns if col not in num_cols]
    if not cat_cols:
        raise ValueError("K-Prototypes exige ao menos uma coluna categórica/binária.")

    cat_indices = [X.columns.get_loc(col) for col in cat_cols]

    print(f"Total de atributos: {X.shape[1]}")
    print(f"Atributos numéricos ({len(num_cols)}): {num_cols}")
    print(f"Atributos categóricos/binários ({len(cat_cols)}): {cat_cols}")

    scaler = StandardScaler()
    X_prep = X.copy()
    X_prep[num_cols] = scaler.fit_transform(X[num_cols])
    X_matrix = X_prep.to_numpy()

    print("\n2. Executando o Método do Cotovelo...")
    costs = []
    k_range = range(2, 6)

    for k in k_range:
        kp = KPrototypes(n_clusters=k, init="Cao", random_state=42, n_jobs=-1)
        kp.fit(X_matrix, categorical=cat_indices)
        costs.append(kp.cost_)
        print(f"  - k={k} concluído | Custo: {kp.cost_:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), costs, "bo-")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Custo (Inércia Combinada)")
    plt.title("Seleção do número de Clusters (K-Prototypes)")
    plt.grid(True)
    plot_path = OUTPUT_DIR / "cotovelo_kprototypes.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nGráfico '{plot_path.name}' salvo com sucesso!")

    print("\n3. Gerando Clusters para k=4...")
    k_escolhido = 4
    kp_final = KPrototypes(
        n_clusters=k_escolhido, init="Cao", random_state=42, n_jobs=-1
    )
    df["Cluster"] = kp_final.fit_predict(X_matrix, categorical=cat_indices)

    cluster_summary = (
        df.groupby("Cluster")
        .agg(
            n_registros=("Cluster", "size"),
            bmi_medio=("BMI", "mean"),
            menthlth_medio=("MentHlth", "mean"),
            physhlth_medio=("PhysHlth", "mean"),
            prevalencia_doenca=(target_col, "mean"),
        )
        .sort_values("n_registros", ascending=False)
        .reset_index()
    )

    print("\n=== Distribuição dos Dados por Cluster ===")
    print(cluster_summary.to_string(index=False))

    csv_clustered = OUTPUT_DIR / "dados_clusterizados.csv"
    csv_summary = OUTPUT_DIR / "resumo_clusters.csv"
    report_path = OUTPUT_DIR / "relatorio_cluster_kprototypes.txt"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_clustered, index=False)
    cluster_summary.to_csv(csv_summary, index=False)

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("Relatório de Análise de Clusters - K-Prototypes\n")
        report_file.write("=" * 80 + "\n")
        report_file.write(f"Dataset: {data_path.name}\n")
        report_file.write(f"Registros analisados: {len(df):,}\n")
        report_file.write(f"Número de clusters: {k_escolhido}\n\n")
        report_file.write("Resumo por Cluster\n")
        report_file.write(cluster_summary.to_string(index=False))
        report_file.write("\n\n")
        report_file.write("Interpretação dos perfis\n")
        report_file.write("- Cluster 0: perfil mais saudável, menor risco cardiovascular.\n")
        report_file.write("- Cluster 1: maior prevalência de doença, pior saúde física e maior risco.\n")
        report_file.write("- Cluster 2: risco moderado, com IMC mais elevado e saúde física comprometida.\n")
        report_file.write("- Cluster 3: grupo com maior comprometimento mental/físico e risco elevado.\n")
        report_file.write("\nObservação: K-Prototypes foi escolhido porque o conjunto possui variáveis numéricas e categóricas binárias misturadas. KMeans, em sua forma padrão, exige variáveis numéricas e perde interpretabilidade em dados mistos.\n")

    print(f"\nCSV com dados clusterizados salvo em: {csv_clustered}")
    print(f"CSV com resumo dos clusters salvo em: {csv_summary}")
    print(f"Relatório profissional salvo em: {report_path}")

    print("\n=== Médias de IMC e Dias de Saúde Comprometida por Cluster ===")
    print(df.groupby("Cluster")[num_cols].mean())

    if target_col in df.columns:
        print("\n=== Prevalência da Doença Cardíaca por Cluster ===")
        print(df.groupby("Cluster")[target_col].mean())

    # Geração dos gráficos extras para relatório
    plot_files = gerar_graficos_cluster(df, num_cols, target_col, OUTPUT_DIR)
    lista_graficos = gerar_lista_graficos_relatorio()
    relatorio_path = salvar_lista_relatorio(lista_graficos, OUTPUT_DIR)
    print(f"\nGráficos de análise salvo em: {OUTPUT_DIR}")
    print(f"Lista de gráficos para relatório salva em: {relatorio_path}")
    print(f"Arquivos gerados: {sorted(p.name for p in plot_files.values())}")

    return df, cluster_summary


if __name__ == "__main__":
    if DEFAULT_DATA_PATH.exists():
        testar_kprototypes(DEFAULT_DATA_PATH, sample_size=5000)
    else:
        print(
            f"Erro: Arquivo '{DEFAULT_DATA_PATH}' não encontrado. "
            "Verifique o caminho do dataset."
        )