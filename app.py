"""Dashboard HeartHealth — layout e orquestração das abas.

As figuras Plotly ficam em `src/dashboard_comps.py`; este arquivo é
responsável apenas pelo carregamento dos dados, KPIs e layout geral.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_comps import (
    fig_box_bmi,
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


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"Arquivo tratado não encontrado em {DATA_PATH}. Execute primeiro o script 'src/data_prep.py'."
        )
        st.stop()
    return pd.read_parquet(DATA_PATH)


# Carregar dados
df = load_data()

# Cabeçalho Principal
st.title("❤️ HeartHealth — Painel de Análise Exploratória e Triagem")
st.markdown(
    "Ferramenta de análise epidemiológica e suporte à triagem de risco clínico cardiovascular."
)

# Abas do Dashboard
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Visão Geral & Epidemiologia",
        "🔍 Outliers & Qualidade dos Dados",
        "🩺 Fatores Clínicos & Triagem",
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
