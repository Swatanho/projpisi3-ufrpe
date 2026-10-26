"""Componentes visuais (figuras Plotly) do dashboard HeartHealth.

Cada função recebe o DataFrame tratado e retorna uma figura Plotly,
mantendo este módulo separado do layout definido em `app.py`.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Prevalência de DCV por cluster, com linha da base e lift anotado."""
    prev_base = dfc["HeartDiseaseorAttack"].mean() * 100
    prev = dfc.groupby("Cluster")["HeartDiseaseorAttack"].mean().reset_index(name="taxa")
    prev["Prevalência (%)"] = prev["taxa"] * 100
    prev["Lift"] = prev["Prevalência (%)"] / prev_base
    prev["Rotulo"] = prev.apply(
        lambda r: f"{r['Prevalência (%)']:.1f}% ({r['Lift']:.2f}x)", axis=1
    )
    fig = px.bar(
        prev,
        x="Cluster",
        y="Prevalência (%)",
        text="Rotulo",
        title="Prevalência de DCV por Cluster (% e lift vs. base)",
        color_discrete_sequence=[COR_NEGATIVA],
    )
    fig.add_hline(
        y=prev_base,
        line_dash="dash",
        line_color="#333333",
        annotation_text=f"Base: {prev_base:.1f}%",
        annotation_position="top left",
    )
    return fig


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


def fig_cluster_composicao_clinica(dfc: pd.DataFrame):
    """% de fatores clínicos de risco por cluster (barras agrupadas)."""
    fatores = ["HighBP", "HighChol", "Smoker", "Stroke", "DiffWalk"]
    comp = dfc.groupby("Cluster")[fatores].mean().mul(100).reset_index()
    df_melt = comp.melt(id_vars="Cluster", var_name="Fator", value_name="%")
    rotulos = {
        "HighBP": "Hipertensão",
        "HighChol": "Colesterol Alto",
        "Smoker": "Fumante",
        "Stroke": "AVC",
        "DiffWalk": "Dificuldade p/ Andar",
    }
    df_melt["Fator"] = df_melt["Fator"].map(rotulos)
    return px.bar(
        df_melt,
        x="Fator",
        y="%",
        color="Cluster",
        barmode="group",
        title="Composição Clínica por Cluster (% com o fator)",
    )


def fig_cluster_radar(resumo: pd.DataFrame):
    """Radar com o perfil dos clusters (eixos escalados 0-1 entre clusters).

    Idade média e GenHlth entram de propósito: permitem checar se o
    gradiente de risco não é artefato de confundidores (ex.: clusters
    apenas mais velhos).
    """
    eixos = {
        "IMC": "bmi_medio",
        "GenHlth": "genhlth_medio",
        "Dias Saúde Física": "physhlth_medio",
        "Dias Saúde Mental": "menthlth_medio",
        "Idade Média": "idade_media",
        "% Hipertensos": "pct_highbp",
        "% Fumantes": "pct_smoker",
    }
    labels = list(eixos)
    perfis = resumo[list(eixos.values())].copy()
    perfis.columns = labels
    amplitude = (perfis.max() - perfis.min()).replace(0, 1)
    norm = (perfis - perfis.min()) / amplitude

    fig = go.Figure()
    for i, (_, linha) in enumerate(norm.iterrows()):
        fig.add_trace(
            go.Scatterpolar(
                r=list(linha) + [linha[labels[0]]],  # fecha o polígono
                theta=labels + [labels[0]],
                fill="toself",
                name=f"Cluster {int(resumo['Cluster'].iloc[i])}",
            )
        )
    fig.update_layout(
        title="Perfil dos Clusters (escalado 0-1 por eixo)",
        polar={"radialaxis": {"range": [0, 1]}},
    )
    return fig


def fig_cluster_cotovelo(metricas: pd.DataFrame):
    """Custo (cotovelo) e silhueta por k, destacando o k escolhido."""
    k_escolhido = int(metricas.loc[metricas["silhueta"].idxmax(), "k"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=metricas["k"],
            y=metricas["custo"],
            name="Custo (cotovelo)",
            mode="lines+markers",
            line={"color": COR_DESTAQUE},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=metricas["k"],
            y=metricas["silhueta"],
            name="Silhueta",
            mode="lines+markers",
            line={"color": COR_POSITIVA},
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=k_escolhido,
        line_dash="dash",
        annotation_text=f"k escolhido = {k_escolhido}",
    )
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_yaxes(title_text="Custo (K-Prototypes)", secondary_y=False)
    fig.update_yaxes(title_text="Silhueta", secondary_y=True)
    fig.update_layout(title="Escolha do k: Cotovelo (custo) e Silhueta")
    return fig
