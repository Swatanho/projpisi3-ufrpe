from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
        fig_dcv = px.pie(
            df,
            names="HeartDiseaseorAttack",
            title="Proporção de Casos (0 = Sem DCV, 1 = Com DCV)",
            color_discrete_sequence=["#2ec4b6", "#e71d36"],
            hole=0.4,
        )
        st.plotly_chart(fig_dcv, use_container_width=True)

    with g_col2:
        st.subheader("DCV por Faixa Etária")
        df_age = (
            df.groupby(["Age", "HeartDiseaseorAttack"])
            .size()
            .reset_transform if False else df.groupby(["Age", "HeartDiseaseorAttack"]).size().reset_index(name="Contagem")
        )
        fig_age = px.bar(
            df_age,
            x="Age",
            y="Contagem",
            color="HeartDiseaseorAttack",
            barmode="group",
            labels={"Age": "Faixa Etária (1 a 13)", "Contagem": "Nº de Pacientes"},
            color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
        )
        st.plotly_chart(fig_age, use_container_width=True)

# ==============================================================================
# ABA 2: OUTLIERS & QUALIDADE DOS DADOS
# ==============================================================================
with tab2:
    st.header("Diagnóstico do Tratamento de Dados e Outliers")

    o_col1, o_col2 = st.columns(2)

    with o_col1:
        st.subheader("Distribuição do IMC (BMI) Pós-Winsorização")
        fig_box = px.box(
            df,
            y="BMI",
            points=False,
            title="Boxplot do IMC (Limites Percentis Aplicados)",
            color_discrete_sequence=["#ff9f1c"],
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with o_col2:
        st.subheader("Outliers Multivariados (Isolation Forest)")
        outliers_count = df["is_outlier"].sum()
        st.write(
            f"**Total de registros sinalizados como anomalia:** {outliers_count:,} ({outliers_count/len(df)*100:.2f}%)"
        )

        fig_out = px.scatter(
            df.sample(5000, random_state=42),  # Amostragem para performance
            x="BMI",
            y="PhysHlth",
            color="is_outlier",
            title="Amostra de Detecção (IMC vs. Dias de Saúde Física Comprometida)",
            color_discrete_map={False: "#2ec4b6", True: "#e71d36"},
            labels={"is_outlier": "É Outlier?"},
        )
        st.plotly_chart(fig_out, use_container_width=True)

# ==============================================================================
# ABA 3: FATORES CLÍNICOS & TRIAGEM
# ==============================================================================
with tab3:
    st.header("Análise de Fatores Clínicos de Risco")

    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.subheader("Prevalência de DCV por Índice de Comorbidades")
        df_com = (
            df.groupby("Comorbidity_Index")["HeartDiseaseorAttack"]
            .mean()
            .reset_index()
        )
        df_com["Taxa_DCV_%"] = df_com["HeartDiseaseorAttack"] * 100

        fig_com = px.bar(
            df_com,
            x="Comorbidity_Index",
            y="Taxa_DCV_%",
            text_auto=".1f",
            title="Taxa de DCV (%) vs. Nº de Comorbidades Acumuladas",
            labels={"Comorbidity_Index": "Índice de Comorbidades (0 a 4)"},
            color_discrete_sequence=["#e71d36"],
        )
        st.plotly_chart(fig_com, use_container_width=True)

    with c_col2:
        st.subheader("Carga de Saúde Comprometida")
        fig_burden = px.histogram(
            df,
            x="Health_Burden_Index",
            color="HeartDiseaseorAttack",
            barmode="overlay",
            title="Distribuição do Comprometimento Físico + Mental (Dias no Mês)",
            color_discrete_map={0: "#2ec4b6", 1: "#e71d36"},
        )
        st.plotly_chart(fig_burden, use_container_width=True)