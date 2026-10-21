"""Componentes visuais (figuras Plotly) do dashboard HeartHealth.

Cada função recebe o DataFrame tratado e retorna uma figura Plotly,
mantendo este módulo separado do layout definido em `app.py`.
"""

import pandas as pd
import plotly.express as px

# Paleta padrão do projeto
COR_POSITIVA = "#2ec4b6"  # sem DCV / registro normal
COR_NEGATIVA = "#e71d36"  # com DCV / anomalia
COR_DESTAQUE = "#ff9f1c"  # atenção


def fig_distribuicao_dcv(df: pd.DataFrame):
    """Rosca com a proporção do desfecho (0 = sem DCV, 1 = com DCV)."""
    return px.pie(
        df,
        names="HeartDiseaseorAttack",
        title="Proporção de Casos (0 = Sem DCV, 1 = Com DCV)",
        color_discrete_sequence=[COR_POSITIVA, COR_NEGATIVA],
        hole=0.4,
    )


def fig_dcv_por_idade(df: pd.DataFrame):
    """Barras agrupadas com o nº de pacientes por faixa etária e desfecho."""
    df_age = (
        df.groupby(["Age", "HeartDiseaseorAttack"]).size().reset_index(name="Contagem")
    )
    return px.bar(
        df_age,
        x="Age",
        y="Contagem",
        color="HeartDiseaseorAttack",
        barmode="group",
        labels={"Age": "Faixa Etária (1 a 13)", "Contagem": "Nº de Pacientes"},
        color_discrete_map={0: COR_POSITIVA, 1: COR_NEGATIVA},
    )


def fig_box_bmi(df: pd.DataFrame):
    """Boxplot do IMC após a winsorização (limites percentis aplicados)."""
    return px.box(
        df,
        y="BMI",
        points=False,
        title="Boxplot do IMC (Limites Percentis Aplicados)",
        color_discrete_sequence=[COR_DESTAQUE],
    )


def fig_outliers_iforest(df: pd.DataFrame, n_amostra: int = 5000, seed: int = 42):
    """Dispersão (IMC vs. dias de saúde física comprometida) destacando outliers do Isolation Forest."""
    amostra = df.sample(n_amostra, random_state=seed)  # amostragem para performance
    return px.scatter(
        amostra,
        x="BMI",
        y="PhysHlth",
        color="is_outlier",
        title="Amostra de Detecção (IMC vs. Dias de Saúde Física Comprometida)",
        color_discrete_map={False: COR_POSITIVA, True: COR_NEGATIVA},
        labels={"is_outlier": "É Outlier?"},
    )


def fig_prevalencia_comorbidade(df: pd.DataFrame):
    """Barras com a taxa de DCV (%) por número de comorbidades acumuladas."""
    df_com = (
        df.groupby("Comorbidity_Index")["HeartDiseaseorAttack"].mean().reset_index()
    )
    df_com["Taxa_DCV_%"] = df_com["HeartDiseaseorAttack"] * 100
    return px.bar(
        df_com,
        x="Comorbidity_Index",
        y="Taxa_DCV_%",
        text_auto=".1f",
        title="Taxa de DCV (%) vs. Nº de Comorbidades Acumuladas",
        labels={"Comorbidity_Index": "Índice de Comorbidades (0 a 4)"},
        color_discrete_sequence=[COR_NEGATIVA],
    )


def fig_health_burden(df: pd.DataFrame):
    """Histograma do índice de carga de saúde (física + mental) por desfecho."""
    return px.histogram(
        df,
        x="Health_Burden_Index",
        color="HeartDiseaseorAttack",
        barmode="overlay",
        title="Distribuição do Comprometimento Físico + Mental (Dias no Mês)",
        color_discrete_map={0: COR_POSITIVA, 1: COR_NEGATIVA},
    )


# ---------------------------------------------------------------------------
# Segmentação por clusters (K-Prototypes) — saídas de src/clustering_teste.py
# ---------------------------------------------------------------------------


def fig_cluster_distribuicao(dfc: pd.DataFrame):
    """Barras com o tamanho de cada cluster (registros e % da base)."""
    contagens = dfc["Cluster"].value_counts().sort_index().reset_index()
    contagens.columns = ["Cluster", "Registros"]
    contagens["% da Base"] = contagens["Registros"] / len(dfc) * 100
    contagens["Rotulo"] = contagens.apply(
        lambda r: f"{r['Registros']:,} ({r['% da Base']:.1f}%)", axis=1
    )
    return px.bar(
        contagens,
        x="Cluster",
        y="Registros",
        text="Rotulo",
        title="Tamanho dos Clusters (Registros e % da Base)",
        labels={"Registros": "Nº de Registros"},
        color_discrete_sequence=[COR_DESTAQUE],
    )


def fig_cluster_prevalencia(dfc: pd.DataFrame):
    """Barras com a prevalência de DCV (%) por cluster."""
    prev = (
        dfc.groupby("Cluster")["HeartDiseaseorAttack"]
        .mean()
        .mul(100)
        .reset_index(name="Prevalência (%)")
    )
    return px.bar(
        prev,
        x="Cluster",
        y="Prevalência (%)",
        text_auto=".1f",
        title="Prevalência de DCV por Cluster (%)",
        color_discrete_sequence=[COR_NEGATIVA],
    )


def fig_cluster_boxplots(dfc: pd.DataFrame):
    """Boxplots das contínuas (IMC e dias de saúde física/mental) por cluster."""
    df_melt = dfc.melt(
        id_vars="Cluster",
        value_vars=["BMI", "MentHlth", "PhysHlth"],
        var_name="Variável",
        value_name="Valor",
    )
    return px.box(
        df_melt,
        x="Cluster",
        y="Valor",
        facet_col="Variável",
        points=False,
        title="Contínuas por Cluster (IMC e Dias de Saúde Comprometida)",
    )
