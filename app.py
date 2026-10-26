"""Dashboard HeartHealth — layout e orquestração das abas.

As figuras Plotly ficam em `src/dashboard_comps.py`; este arquivo é
responsável apenas pelo carregamento dos dados, KPIs e layout geral.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_comps import (
    fig_box_bmi,
    fig_cluster_boxplots,
    fig_cluster_composicao_clinica,
    fig_cluster_cotovelo,
    fig_cluster_distribuicao,
    fig_cluster_prevalencia,
    fig_cluster_radar,
    fig_dcv_por_idade,
    fig_distribuicao_dcv,
    fig_health_burden,
    fig_outliers_iforest,
    fig_prevalencia_comorbidade,
)

# Configuração da página
st.set_page_config(
    page_title="HeartHealth - Dashboard de Triagem e Análise",
    page_icon="❤️",
    layout="wide",
)

# Caminho da base tratada
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tratados" / "diabetes_processed.parquet"
RESULTADOS_CLUSTER_PATH = BASE_DIR / "data" / "resultados_cluster"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"Arquivo tratado não encontrado em {DATA_PATH}. Execute primeiro o script 'src/data_prep.py'."
        )
        st.stop()
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_cluster_resultados():
    """Carrega as saídas do clustering (src/clustering_teste.py); None se ausentes."""
    parquet = RESULTADOS_CLUSTER_PATH / "dados_clusterizados.parquet"
    resumo = RESULTADOS_CLUSTER_PATH / "resumo_clusters.csv"
    metricas = RESULTADOS_CLUSTER_PATH / "metricas_k.csv"
    if not (parquet.exists() and resumo.exists() and metricas.exists()):
        return None
    return {
        "dados": pd.read_parquet(parquet),
        "resumo": pd.read_csv(resumo),
        "metricas": pd.read_csv(metricas),
    }


# Carregar dados
df = load_data()

# Cabeçalho Principal
st.title("❤️ HeartHealth — Painel de Análise Exploratória e Triagem")
st.markdown(
    "Ferramenta de análise epidemiológica e suporte à triagem de risco clínico cardiovascular."
)

# Abas do Dashboard
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visão Geral & Epidemiologia",
        "🔍 Outliers & Qualidade dos Dados",
        "🩺 Fatores Clínicos & Triagem",
        "🧩 Segmentação por Clusters",
    ]
)

# ==============================================================================
# ABA 1: VISÃO GERAL & EPIDEMIOLOGIA
# ==============================================================================
with tab1:
    st.header("Panorama Geral da Base de Dados")

    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    total_pacientes = len(df)
    casos_dcv = df["HeartDiseaseorAttack"].sum()
    pct_dcv = (casos_dcv / total_pacientes) * 100
    imc_medio = df["BMI"].mean()

    col1.metric("Total de Pacientes", f"{total_pacientes:,}")
    col2.metric("Casos de Doença Cardíaca", f"{casos_dcv:,}")
    col3.metric("Prevalência de DCV", f"{pct_dcv:.2f}%")
    col4.metric("IMC Médio (Tratado)", f"{imc_medio:.1f}")

    st.markdown("---")

    # Gráficos de Distribuição
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.subheader("Distribuição do Desfecho (DCV)")
        st.plotly_chart(fig_distribuicao_dcv(df), use_container_width=True)

    with g_col2:
        st.subheader("DCV por Faixa Etária")
        st.plotly_chart(fig_dcv_por_idade(df), use_container_width=True)

# ==============================================================================
# ABA 2: OUTLIERS & QUALIDADE DOS DADOS
# ==============================================================================
with tab2:
    st.header("Diagnóstico do Tratamento de Dados e Outliers")

    o_col1, o_col2 = st.columns(2)

    with o_col1:
        st.subheader("Distribuição do IMC (BMI) Pós-Winsorização")
        st.plotly_chart(fig_box_bmi(df), use_container_width=True)

    with o_col2:
        st.subheader("Outliers Multivariados (Isolation Forest)")
        outliers_count = df["is_outlier"].sum()
        st.write(
            f"**Total de registros sinalizados como anomalia:** {outliers_count:,} ({outliers_count/len(df)*100:.2f}%)"
        )
        st.plotly_chart(fig_outliers_iforest(df), use_container_width=True)

# ==============================================================================
# ABA 3: FATORES CLÍNICOS & TRIAGEM
# ==============================================================================
with tab3:
    st.header("Análise de Fatores Clínicos de Risco")

    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.subheader("Prevalência de DCV por Índice de Comorbidades")
        st.plotly_chart(fig_prevalencia_comorbidade(df), use_container_width=True)

    with c_col2:
        st.subheader("Carga de Saúde Comprometida")
        st.plotly_chart(fig_health_burden(df), use_container_width=True)

# ==============================================================================
# ABA 4: SEGMENTAÇÃO POR CLUSTERS
# ==============================================================================
with tab4:
    st.header("Segmentação de Perfis (K-Prototypes)")

    resultados = load_cluster_resultados()
    if resultados is None:
        st.info(
            "Resultados do clustering não encontrados em `data/resultados_cluster/`. "
            "Execute o script `src/clustering_teste.py` para gerar os dados desta aba."
        )
    else:
        dfc = resultados["dados"]
        resumo = resultados["resumo"]
        metricas = resultados["metricas"]

        # KPIs: k escolhido (maior silhueta), tamanho da base, silhueta e lift
        melhor_k = metricas.loc[metricas["silhueta"].idxmax()]
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        k_col1.metric("Clusters (k)", int(melhor_k["k"]))
        k_col2.metric("Registros Segmentados", f"{len(dfc):,}")
        k_col3.metric("Silhueta do k Escolhido", f"{melhor_k['silhueta']:.3f}")
        k_col4.metric(
            "Lift do Cluster de Maior Risco", f"{resumo['lift_dcv'].max():.2f}x"
        )

        st.markdown("---")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.subheader("Tamanho dos Clusters")
            st.plotly_chart(fig_cluster_distribuicao(dfc), use_container_width=True)

        with s_col2:
            st.subheader("Prevalência de DCV por Cluster")
            st.plotly_chart(fig_cluster_prevalencia(dfc), use_container_width=True)

        st.subheader("Contínuas por Cluster")
        st.plotly_chart(fig_cluster_boxplots(dfc), use_container_width=True)

        cl_col1, cl_col2 = st.columns(2)
        with cl_col1:
            st.subheader("Composição Clínica por Cluster")
            st.plotly_chart(
                fig_cluster_composicao_clinica(dfc), use_container_width=True
            )

        with cl_col2:
            st.subheader("Perfil dos Clusters (Radar)")
            st.plotly_chart(fig_cluster_radar(resumo), use_container_width=True)
            st.caption(
                "Idade média e GenHlth no radar permitem checar se o gradiente "
                "de risco é real ou artefato de idade/percepção de saúde."
            )

        with st.expander("Método: escolha do k (cotovelo + silhueta)"):
            st.plotly_chart(fig_cluster_cotovelo(metricas), use_container_width=True)
            st.caption(
                "O k final é o de maior silhueta (calculada sobre contínuas "
                "padronizadas + one-hot das categóricas); a curva de custo "
                "(cotovelo) é exibida como apoio à decisão."
            )
