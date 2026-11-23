"""Componentes visuais (figuras Plotly) do dashboard HeartHealth.

Cada função recebe o DataFrame tratado e retorna uma figura Plotly,
mantendo este módulo separado do layout definido em `app.py`.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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


# ---------------------------------------------------------------------------
# Comparação Isolation Forest x HDBSCAN — saídas de src/outliers_hdbscan.py
# ---------------------------------------------------------------------------


def _amostra_scatter(df: pd.DataFrame, n_amostra: int, seed: int) -> pd.DataFrame:
    """Amostra aleatória (se necessário) para manter o scatter responsivo."""
    if len(df) > n_amostra:
        return df.sample(n_amostra, random_state=seed)
    return df


def fig_outliers_scatter_comparacao(
    df: pd.DataFrame, coluna: str, titulo: str, n_amostra: int = 5000, seed: int = 42
):
    """Scatter reutilizável (IMC × dias de saúde física) colorido pela flag de um método.

    `coluna` é booleana (`is_outlier` p/ Isolation Forest ou `hdbscan` p/ o ruído
    do HDBSCAN). Para o HDBSCAN, `df` já deve estar filtrado pela subamostra
    avaliada (`hdbscan_avaliado == True`).
    """
    amostra = _amostra_scatter(df, n_amostra, seed)
    return px.scatter(
        amostra,
        x="BMI",
        y="PhysHlth",
        color=coluna,
        title=titulo,
        color_discrete_map={False: COR_POSITIVA, True: COR_NEGATIVA},
        labels={"PhysHlth": "Dias de Saúde Física Comprometida", coluna: "É Outlier?"},
    )


def fig_outliers_discordancia(df_subs: pd.DataFrame, n_amostra: int = 5000, seed: int = 42):
    """Dispersão da concordância/discordância IF × HDBSCAN (subamostra avaliada).

    Categoriza cada registro em: ambos marcaram como outlier, só o Isolation
    Forest, só o HDBSCAN, ou nenhum — evidenciando onde os métodos divergem.
    """
    amostra = _amostra_scatter(df_subs, n_amostra, seed).copy()
    iso = amostra["is_outlier"].astype(bool)
    hdb = amostra["hdbscan"].astype(bool)
    amostra["Categoria"] = np.select(
        [iso & hdb, iso & ~hdb, ~iso & hdb],
        ["Ambos outlier", "Só Isolation Forest", "Só HDBSCAN"],
        default="Nenhum",
    )
    cores = {
        "Ambos outlier": COR_NEGATIVA,
        "Só Isolation Forest": COR_DESTAQUE,
        "Só HDBSCAN": "#5b8def",
        "Nenhum": COR_POSITIVA,
    }
    ordem = ["Nenhum", "Só HDBSCAN", "Só Isolation Forest", "Ambos outlier"]
    return px.scatter(
        amostra,
        x="BMI",
        y="PhysHlth",
        color="Categoria",
        title="Concordância/Discordância entre Isolation Forest e HDBSCAN",
        color_discrete_map=cores,
        category_orders={"Categoria": ordem},
        labels={"PhysHlth": "Dias de Saúde Física Comprometida"},
    )


def fig_outliers_lift(df: pd.DataFrame):
    """Barras do lift de DCV (prevalência nos outliers ÷ base) por método.

    O Isolation Forest é medido na base completa; o HDBSCAN apenas na
    subamostra avaliada (coluna `hdbscan_avaliado`).
    """
    prev_base = df["HeartDiseaseorAttack"].mean()
    subs = df[df["hdbscan_avaliado"]]
    lift_if = df.loc[df["is_outlier"], "HeartDiseaseorAttack"].mean() / prev_base
    lift_hdb = subs.loc[subs["hdbscan"], "HeartDiseaseorAttack"].mean() / prev_base
    dados = pd.DataFrame(
        [
            {"Método": "Isolation Forest", "Lift": lift_if},
            {"Método": "HDBSCAN (subamostra)", "Lift": lift_hdb},
        ]
    )
    fig = px.bar(
        dados,
        x="Método",
        y="Lift",
        text="Lift",
        title="Lift de DCV nos Outliers (prevalência nos outliers ÷ base)",
        color="Método",
        color_discrete_map={
            "Isolation Forest": COR_NEGATIVA,
            "HDBSCAN (subamostra)": COR_DESTAQUE,
        },
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#333333",
        annotation_text="Lift = 1 (sem sinal)",
    )
    fig.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


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
# Segmentação por clusters (K-Prototypes) — saídas de src/clustering_comparacao.py
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


def fig_cluster_cotovelo(metricas: pd.DataFrame, k_escolhido=None):
    """Custo (cotovelo) e silhueta por k, destacando o k escolhido."""
    if k_escolhido is None:
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


def fig_cluster_gradiente(comparacao: pd.DataFrame, k_final: int):
    """Spread de risco e lift máximo por k (K-Prototypes), com o k escolhido.

    Mostra que o gradiente de risco cresce de k=2 até ~k=4 e então estabiliza,
    justificando k>2 mesmo com a silhueta favorecendo k=2.
    """
    kp = comparacao[comparacao["estrategia"] == "K-Prototypes"].copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=kp["k"], y=kp["spread"], name="Spread de risco",
            mode="lines+markers", line={"color": COR_NEGATIVA},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=kp["k"], y=kp["lift_max"], name="Lift máximo",
            mode="lines+markers", line={"color": COR_DESTAQUE},
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=k_final, line_dash="dash",
        annotation_text=f"k escolhido = {k_final}",
    )
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_yaxes(title_text="Spread de risco (max−min prev.)", secondary_y=False)
    fig.update_yaxes(title_text="Lift máximo (x prevalência geral)", secondary_y=True)
    fig.update_layout(title="Gradiente de risco por k (K-Prototypes)")
    return fig


def fig_fatores_risco(fatores: list):
    """Barras horizontais com as principais contribuições SHAP de um paciente.

    `fatores` é a lista retornada por `PredictorDCV.prever()['fatores_principais']`
    (contribuições SHAP em escala de logito; vermelho aumenta o risco,
    verde-água reduz).
    """
    dados = [
        {"rotulo": f"{f['feature']} = {f['valor']:g}", "contribuicao": f["contribuicao"]}
        for f in fatores
    ][::-1]  # maior |contribuição| no topo do gráfico
    fig = go.Figure(
        go.Bar(
            x=[d["contribuicao"] for d in dados],
            y=[d["rotulo"] for d in dados],
            orientation="h",
            marker_color=[
                COR_NEGATIVA if d["contribuicao"] > 0 else COR_POSITIVA for d in dados
            ],
            text=[f"{d['contribuicao']:+.2f}" for d in dados],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#999999")
    fig.update_layout(
        title="Fatores que mais influenciaram esta predição (SHAP, escala de logito)",
        xaxis_title="Contribuição para o risco (positivo = aumenta)",
        margin=dict(l=10, r=40, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Auditoria do modelo (vazamento, balanceamento, threshold) — src/auditoria_modelo.py
# ---------------------------------------------------------------------------


def fig_auditoria_sweep(sweep: pd.DataFrame):
    """Recall, precisão e F1 ao longo do threshold, com marcos de triagem e 0.5.

    Evidencia que o gap recall × precisão é um trade-off de threshold sobre
    uma base desbalanceada, não um defeito do modelo.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["recall"], name="Recall",
            mode="lines+markers", line={"color": COR_POSITIVA},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["precisao"], name="Precisão",
            mode="lines+markers", line={"color": COR_NEGATIVA},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["f1"], name="F1",
            mode="lines+markers", line={"color": COR_DESTAQUE, "dash": "dot"},
        )
    )
    fig.add_vline(x=0.36, line_dash="dash", line_color="#666666",
                  annotation_text="triagem (0.36)")
    fig.add_vline(x=0.5, line_dash="dot", line_color="#999999",
                  annotation_text="0.5")
    fig.update_layout(
        title="Recall × Precisão por threshold (modelo completo)",
        xaxis_title="Threshold de decisão",
        yaxis_title="Métrica",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_auditoria_ablacao(ablacao: pd.DataFrame):
    """ROC-AUC e PR-AUC por configuração (modelo completo vs. ablações)."""
    dados = ablacao.copy().sort_values("roc_auc", ascending=False)
    completo = dados["config"].str.startswith("completo")
    cores = [COR_NEGATIVA if c else COR_POSITIVA for c in completo]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dados["config"], y=dados["roc_auc"], name="ROC-AUC",
            marker_color=cores, text=dados["roc_auc"].round(3),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=dados["config"], y=dados["pr_auc"], name="PR-AUC",
            marker_color=COR_DESTAQUE, text=dados["pr_auc"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Ablação: modelo completo vs. sem features suspeitas",
        barmode="group",
        xaxis_title="Configuração",
        yaxis_title="AUC",
        xaxis_tickangle=-20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_auditoria_univariada(univ: pd.DataFrame, top: int = 12):
    """AUC discriminante de cada feature isolada, com o limiar de vazamento."""
    dados = univ.head(top).sort_values("auc_discriminante", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=dados["auc_discriminante"], y=dados["feature"], orientation="h",
            marker_color=COR_NEGATIVA,
            text=dados["auc_discriminante"].round(3), textposition="outside",
        )
    )
    fig.add_vline(
        x=0.85, line_dash="dash", line_color="#333333",
        annotation_text="limiar de vazamento (0.85)",
    )
    fig.update_layout(
        title="Poder discriminante isolado por feature (AUC, no teste)",
        xaxis_title="AUC (quanto mais perto de 1, mais 'espelha' o desfecho)",
        yaxis_title="Feature",
        xaxis=dict(range=[0.45, 0.9]),
    )
    return fig


# ---------------------------------------------------------------------------
# Redução de dimensionalidade (PCA + UMAP) — src/reducao_dimensionalidade.py
# ---------------------------------------------------------------------------


def fig_reducao_scree(pca_explicada: pd.DataFrame):
    """Variância explicada (barras) e acumulada (linha) por componente do PCA."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pca_explicada["componente"], y=pca_explicada["var_explicada"],
            name="Variância explicada", marker_color=COR_POSITIVA,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pca_explicada["componente"], y=pca_explicada["var_acumulada"],
            name="Acumulada", mode="lines+markers", line={"color": COR_NEGATIVA},
        )
    )
    for pct in (0.80, 0.90, 0.95):
        fig.add_hline(
            y=pct, line_dash="dot", line_color="#999999",
            annotation_text=f"{int(pct * 100)}%",
        )
    fig.update_layout(
        title="PCA — variância explicada por componente",
        xaxis_title="Componente",
        yaxis_title="Variância explicada",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_reducao_projecao(proj: pd.DataFrame, x: str, y: str, color: str, titulo: str):
    """Scatter 2D de uma projeção (PCA ou UMAP) colorida por uma coluna."""
    if color == "HeartDiseaseorAttack":
        mapa = {0: COR_POSITIVA, 1: COR_NEGATIVA}
        rotulo = "DCV"
    else:
        mapa = {False: COR_POSITIVA, True: COR_NEGATIVA}
        rotulo = "Ruído HDBSCAN"
    return px.scatter(
        proj, x=x, y=y, color=color, title=titulo,
        color_discrete_map=mapa, labels={color: rotulo}, opacity=0.5,
    )


def fig_reducao_comparacao(comparacao: pd.DataFrame):
    """ROC-AUC e PR-AUC do modelo completo vs. componentes principais."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparacao["config"], y=comparacao["roc_auc"], name="ROC-AUC",
            marker_color=COR_NEGATIVA, text=comparacao["roc_auc"].round(3),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparacao["config"], y=comparacao["pr_auc"], name="PR-AUC",
            marker_color=COR_DESTAQUE, text=comparacao["pr_auc"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Impacto da redução no modelo (20 features vs. PCA)",
        barmode="group",
        xaxis_title="Configuração",
        yaxis_title="AUC",
        xaxis_tickangle=-15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# SMOTE × SMOTENC — src/smote_smotenc.py
# ---------------------------------------------------------------------------


def fig_smote_integridade(integridade: pd.DataFrame):
    """% de células de features binárias que viraram valor fracionário inválido."""
    dados = integridade.copy()
    dados["pct_invalidas"] = dados["frac_binarias_invalidas"] * 100
    cores = [COR_NEGATIVA if m == "SMOTE" else COR_POSITIVA for m in dados["metodo"]]
    fig = go.Figure(
        go.Bar(
            x=dados["metodo"], y=dados["pct_invalidas"],
            marker_color=cores, text=dados["pct_invalidas"].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Integridade dos sintéticos (% de células binárias inválidas)",
        xaxis_title="Método",
        yaxis_title="% de valores fracionários (fora de 0/1)",
        yaxis=dict(range=[0, max(dados["pct_invalidas"]) * 1.25]),
    )
    return fig


def fig_smote_comparacao(resultados: pd.DataFrame):
    """ROC-AUC e PR-AUC por estratégia de balanceamento (no teste)."""
    dados = resultados.copy()
    dados["rotulo"] = dados["estrategia"].replace(
        {"scale_pos_weight (sem oversampling)": "scale_pos_weight"}
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dados["rotulo"], y=dados["roc_auc"], name="ROC-AUC",
            marker_color=COR_NEGATIVA, text=dados["roc_auc"].round(3),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=dados["rotulo"], y=dados["pr_auc"], name="PR-AUC",
            marker_color=COR_DESTAQUE, text=dados["pr_auc"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Métricas por estratégia de balanceamento (teste)",
        barmode="group",
        xaxis_title="Estratégia",
        yaxis_title="AUC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Explicabilidade global (SHAP) — src/shap_explain_pisi3.py
# ---------------------------------------------------------------------------


def fig_shap_importancia(importancia: pd.DataFrame):
    """Barra horizontal da importância global (média do |valor SHAP|)."""
    dados = importancia.sort_values("shap_media_absoluta", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=dados["shap_media_absoluta"], y=dados["feature"], orientation="h",
            marker_color=COR_NEGATIVA,
            text=dados["shap_media_absoluta"].round(3), textposition="outside",
        )
    )
    fig.update_layout(
        title="Importância global (SHAP — média do |valor SHAP|)",
        xaxis_title="Média |SHAP| (impacto no logito)",
        yaxis_title="Feature",
        height=600,
    )
    return fig


def fig_shap_beeswarm(shap_values, X, features, n_amostra: int = 800, seed: int = 42):
    """Beeswarm SHAP: um ponto por paciente, cor = valor da feature.

    `shap_values` (n×m) e `X` (n×m) já devem estar na ordem de importância
    (`features`), do mais ao menos importante.
    """
    rs = np.random.RandomState(seed)
    shap_values = np.asarray(shap_values, dtype=float)
    X = np.asarray(X, dtype=float)
    if shap_values.shape[0] > n_amostra:
        idx = rs.choice(shap_values.shape[0], n_amostra, replace=False)
        shap_values = shap_values[idx]
        X = X[idx]

    n, m = shap_values.shape
    # normaliza cada feature para colorir (0 = baixo, 1 = alto)
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    Xn = (X - xmin) / (xmax - xmin + 1e-9)

    linhas = []
    for j in range(m):
        yj = j + rs.uniform(-0.35, 0.35, n)  # jitter vertical
        for i in range(n):
            linhas.append({
                "feature": features[j],
                "shap": float(shap_values[i, j]),
                "valor_norm": float(Xn[i, j]),
            })
    df = pd.DataFrame(linhas)
    fig = px.scatter(
        df, x="shap", y="feature", color="valor_norm",
        color_continuous_scale="RdBu_r",
        labels={
            "shap": "Valor SHAP (impacto no logito)",
            "valor_norm": "Valor da feature (baixo → alto)",
        },
    )
    fig.add_vline(x=0, line_width=1, line_color="#999999")
    fig.update_layout(
        title="Beeswarm SHAP (cada ponto = um paciente; cor = valor da feature)",
        height=600,
        yaxis=dict(autorange="reversed"),
        coloraxis_colorbar=dict(title="valor"),
    )
    return fig
