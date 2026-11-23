"""Dashboard HeartHealth — layout e orquestração das abas.

As figuras Plotly ficam em `src/dashboard_comps.py`; este arquivo é
responsável apenas pelo carregamento dos dados, KPIs e layout geral.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_comps import (
    fig_auditoria_ablacao,
    fig_auditoria_sweep,
    fig_auditoria_univariada,
    fig_box_bmi,
    fig_cluster_boxplots,
    fig_cluster_composicao_clinica,
    fig_cluster_cotovelo,
    fig_cluster_distribuicao,
    fig_cluster_gradiente,
    fig_cluster_prevalencia,
    fig_cluster_radar,
    fig_dcv_por_idade,
    fig_distribuicao_dcv,
    fig_health_burden,
    fig_outliers_discordancia,
    fig_outliers_iforest,
    fig_outliers_lift,
    fig_outliers_scatter_comparacao,
    fig_prevalencia_comorbidade,
    fig_reducao_comparacao,
    fig_reducao_projecao,
    fig_reducao_scree,
    fig_shap_beeswarm,
    fig_shap_importancia,
    fig_smote_comparacao,
    fig_smote_integridade,
)

from src.simulador import render_aba_simulador

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
    """Carrega as saídas do clustering (src/clustering_comparacao.py); None se ausentes."""
    parquet = RESULTADOS_CLUSTER_PATH / "dados_clusterizados.parquet"
    resumo = RESULTADOS_CLUSTER_PATH / "resumo_clusters.csv"
    metricas = RESULTADOS_CLUSTER_PATH / "metricas_k.csv"
    comparacao = RESULTADOS_CLUSTER_PATH / "metricas_comparacao.csv"
    if not (parquet.exists() and resumo.exists() and metricas.exists()):
        return None
    dados = {
        "dados": pd.read_parquet(parquet),
        "resumo": pd.read_csv(resumo),
        "metricas": pd.read_csv(metricas),
    }
    if comparacao.exists():
        dados["comparacao"] = pd.read_csv(comparacao)
    else:
        dados["comparacao"] = None
    return dados


@st.cache_data
def load_outlier_flags():
    """Carrega as flags da comparação IF × HDBSCAN (src/outliers_hdbscan.py).

    Retorna um DataFrame alinhado por posição com a base tratada, com as
    colunas `if_20cols`, `hdbscan`, `hdbscan_avaliado` e `glosh`; None se
    o arquivo ainda não foi gerado.
    """
    flags_path = BASE_DIR / "data" / "outliers" / "flags_outliers.parquet"
    if not flags_path.exists():
        return None
    return pd.read_parquet(flags_path)


@st.cache_data
def load_auditoria():
    """Carrega as saídas da auditoria de vazamento/balanceamento (src/auditoria_modelo.py)."""
    base = BASE_DIR / "data" / "modelo"
    paths = {
        "univ": base / "auditoria_univariada.csv",
        "ablacao": base / "ablacao_leakage.csv",
        "sweep": base / "sweep_threshold.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {chave: pd.read_csv(p) for chave, p in paths.items()}


@st.cache_data
def load_reducao():
    """Carrega as saídas da redução de dimensionalidade (src/reducao_dimensionalidade.py)."""
    base = BASE_DIR / "data" / "reducao"
    paths = {
        "pca": base / "pca_explicada.csv",
        "comparacao": base / "comparacao_reducao.csv",
        "proj_pca": base / "pca_projecao.parquet",
        "proj_umap": base / "umap_projecao.parquet",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {
        "pca": pd.read_csv(paths["pca"]),
        "comparacao": pd.read_csv(paths["comparacao"]),
        "proj_pca": pd.read_parquet(paths["proj_pca"]),
        "proj_umap": pd.read_parquet(paths["proj_umap"]),
    }


@st.cache_data
def load_smote():
    """Carrega as saídas da comparação SMOTE × SMOTENC (src/smote_smotenc.py)."""
    base = BASE_DIR / "data" / "modelo"
    paths = {
        "resultados": base / "smote_smotenc.csv",
        "integridade": base / "integridade_sinteticos.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {chave: pd.read_csv(p) for chave, p in paths.items()}


# Ordem das features no contrato == ordem das colunas de shap_valores.npy/shap_X.npy
FEATURES_ORDEM = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
]


@st.cache_data
def load_shap():
    """Carrega as saídas do SHAP global (src/shap_explain_pisi3.py).

    Reordena as colunas de shap_valores/shap_X para a ordem de importância.
    """
    base = BASE_DIR / "data" / "modelo"
    paths = {
        "importancia": base / "shap_importancia.csv",
        "valores": base / "shap_valores.npy",
        "X": base / "shap_X.npy",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    importancia = pd.read_csv(paths["importancia"])
    shap = np.load(paths["valores"])
    X = np.load(paths["X"])
    ordem = importancia["feature"].tolist()  # já em ordem decrescente de importância
    col_idx = [FEATURES_ORDEM.index(f) for f in ordem]
    return {
        "importancia": importancia,
        "shap": shap[:, col_idx],
        "X": X[:, col_idx],
        "features": ordem,
    }


# Carregar dados
df = load_data()

# Cabeçalho Principal
st.title("❤️ HeartHealth — Painel de Análise Exploratória e Triagem")
st.markdown(
    "Ferramenta de análise epidemiológica e suporte à triagem de risco clínico cardiovascular."
)

# Abas do Dashboard — radio horizontal: a seleção persiste entre reruns
# (st.tabs sempre volta para a primeira aba após qualquer interação).
ABAS = [
    "📊 Visão Geral & Epidemiologia",
    "🔍 Outliers & Qualidade dos Dados",
    "🩺 Fatores Clínicos & Triagem",
    "🧩 Segmentação por Clusters",
    "🧪 Simulador de Risco (Modelo)",
]
aba = st.radio("Navegação", ABAS, horizontal=True, label_visibility="collapsed")

# ==============================================================================
# ABA 1: VISÃO GERAL & EPIDEMIOLOGIA
# ==============================================================================
if aba == ABAS[0]:
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

    # --- Redução de dimensionalidade (PCA + UMAP) ---------------------------
    st.markdown("---")
    st.subheader("Redução de Dimensionalidade (PCA + UMAP)")

    reducao = load_reducao()
    if reducao is None:
        st.info(
            "Resultados de redução de dimensionalidade não encontrados em "
            "`data/reducao/`. Execute o script `src/reducao_dimensionalidade.py`."
        )
    else:
        pca = reducao["pca"]
        comparacao = reducao["comparacao"]
        proj_pca = reducao["proj_pca"]
        proj_umap = reducao["proj_umap"]

        k90 = int((pca["var_acumulada"] >= 0.90).idxmax()) + 1
        roc_full = comparacao.loc[comparacao["n_dims"] == 20, "roc_auc"].iloc[0]
        roc_pca = comparacao.loc[
            comparacao["config"].str.startswith("PCA"), "roc_auc"
        ].max()

        r1, r2, r3 = st.columns(3)
        r1.metric("Componentes p/ 90% da variância", k90)
        r2.metric("ROC-AUC (20 features)", f"{roc_full:.3f}")
        r3.metric("ROC-AUC (PCA reduzido)", f"{roc_pca:.3f}")

        st.plotly_chart(fig_reducao_scree(pca), use_container_width=True)

        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                fig_reducao_projecao(
                    proj_pca, "PC1", "PC2", "HeartDiseaseorAttack",
                    "PCA 2D — colorido por DCV",
                ),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                fig_reducao_projecao(
                    proj_umap, "UMAP1", "UMAP2", "HeartDiseaseorAttack",
                    "UMAP 2D — colorido por DCV",
                ),
                use_container_width=True,
            )

        if "hdbscan_ruido" in proj_umap.columns:
            st.plotly_chart(
                fig_reducao_projecao(
                    proj_umap, "UMAP1", "UMAP2", "hdbscan_ruido",
                    "UMAP 2D — colorido por ruído do HDBSCAN",
                ),
                use_container_width=True,
            )

        st.plotly_chart(fig_reducao_comparacao(comparacao), use_container_width=True)
        st.caption(
            "A redução serve para visualização e entendimento da estrutura, não "
            "para ganho preditivo: o modelo com as 20 features originais iguala "
            "ou supera os componentes principais."
        )

# ==============================================================================
# ABA 2: OUTLIERS & QUALIDADE DOS DADOS
# ==============================================================================
if aba == ABAS[1]:
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

    # --- Comparação Isolation Forest × HDBSCAN -------------------------------
    st.markdown("---")
    st.subheader("Comparação: Isolation Forest × HDBSCAN")

    flags = load_outlier_flags()
    if flags is None or len(flags) != len(df):
        st.info(
            "Resultados da comparação HDBSCAN não encontrados em "
            "`data/outliers/flags_outliers.parquet`. Execute o script "
            "`src/outliers_hdbscan.py` para gerar esta seção."
        )
    else:
        df_comp = df.reset_index(drop=True).copy()
        df_comp["hdbscan"] = flags["hdbscan"].to_numpy()
        df_comp["hdbscan_avaliado"] = flags["hdbscan_avaliado"].to_numpy()
        subs = df_comp[df_comp["hdbscan_avaliado"]]

        prev_base = df_comp["HeartDiseaseorAttack"].mean()
        n_if = int(df_comp["is_outlier"].sum())
        n_hdb = int(subs["hdbscan"].sum())
        lift_if = (
            df_comp.loc[df_comp["is_outlier"], "HeartDiseaseorAttack"].mean()
            / prev_base
        )
        lift_hdb = (
            subs.loc[subs["hdbscan"], "HeartDiseaseorAttack"].mean() / prev_base
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Outliers (Isolation Forest)", f"{n_if:,}")
        k2.metric("Ruído (HDBSCAN)", f"{n_hdb:,}")
        k3.metric("Lift IF", f"{lift_if:.2f}x")
        k4.metric("Lift HDBSCAN", f"{lift_hdb:.2f}x")
        st.caption(
            "O HDBSCAN foi avaliado numa subamostra estratificada de 50 mil "
            "registros (rótulo -1 = ruído); o Isolation Forest cobre a base completa."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                fig_outliers_scatter_comparacao(
                    df_comp, "is_outlier", "Isolation Forest — IMC × Dias de Saúde Física"
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                fig_outliers_scatter_comparacao(
                    subs, "hdbscan", "HDBSCAN (ruído) — IMC × Dias de Saúde Física"
                ),
                use_container_width=True,
            )

        st.plotly_chart(fig_outliers_discordancia(subs), use_container_width=True)
        st.plotly_chart(fig_outliers_lift(df_comp), use_container_width=True)

# ==============================================================================
# ABA 3: FATORES CLÍNICOS & TRIAGEM
# ==============================================================================
if aba == ABAS[2]:
    st.header("Análise de Fatores Clínicos de Risco")

    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.subheader("Prevalência de DCV por Índice de Comorbidades")
        st.plotly_chart(fig_prevalencia_comorbidade(df), use_container_width=True)

    with c_col2:
        st.subheader("Carga de Saúde Comprometida")
        st.plotly_chart(fig_health_burden(df), use_container_width=True)

    # --- Auditoria do modelo (vazamento, balanceamento, threshold) ---------
    st.markdown("---")
    st.subheader("Auditoria do Modelo de Triagem (vazamento, balanceamento, threshold)")

    auditoria = load_auditoria()
    if auditoria is None:
        st.info(
            "Resultados da auditoria não encontrados em `data/modelo/`. Execute "
            "o script `src/auditoria_modelo.py` para gerar esta seção."
        )
    else:
        univ = auditoria["univ"]
        ablacao = auditoria["ablacao"]
        sweep = auditoria["sweep"]

        # Destaques (leitura automática dos arquivos)
        top_univ = univ.sort_values("auc_discriminante", ascending=False).iloc[0]
        roc_completo = ablacao.loc[
            ablacao["config"].str.startswith("completo"), "roc_auc"
        ].iloc[0]
        t_050 = sweep.loc[(sweep["threshold"] - 0.5).abs().idxmin()]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("ROC-AUC (teste)", f"{roc_completo:.3f}")
        a2.metric("Recall @0.5", f"{t_050['recall']:.2f}")
        a3.metric("Precisão @0.5", f"{t_050['precisao']:.2f}")
        a4.metric("Feature mais discriminante", f"{top_univ['feature']} ({top_univ['auc_discriminante']:.2f})")

        st.caption(
            "Nenhuma feature isolada atinge o limiar de vazamento (AUC 0.85); "
            "a ablação mostra que o modelo não depende indevidamente de uma "
            "variável só; e o gap recall × precisão decorre do desbalanceamento "
            "(~10%) + threshold de triagem sensível."
        )

        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.plotly_chart(fig_auditoria_sweep(sweep), use_container_width=True)
        with a_col2:
            st.plotly_chart(fig_auditoria_univariada(univ), use_container_width=True)

        st.plotly_chart(fig_auditoria_ablacao(ablacao), use_container_width=True)

    # --- SMOTE × SMOTENC (estratégia de balanceamento) -----------------------
    st.markdown("---")
    st.subheader("SMOTE × SMOTENC (estratégia de balanceamento)")

    smote = load_smote()
    if smote is None:
        st.info(
            "Resultados da comparação SMOTE × SMOTENC não encontrados em "
            "`data/modelo/`. Execute o script `src/smote_smotenc.py`."
        )
    else:
        resultados = smote["resultados"]
        integridade = smote["integridade"]

        roc_base = resultados.loc[
            resultados["estrategia"].str.startswith("scale_pos_weight"), "roc_auc"
        ].iloc[0]
        frac_inv_smote = integridade.loc[
            integridade["metodo"] == "SMOTE", "frac_binarias_invalidas"
        ].iloc[0]
        frac_inv_smotenc = integridade.loc[
            integridade["metodo"] == "SMOTENC", "frac_binarias_invalidas"
        ].iloc[0]

        s1, s2, s3 = st.columns(3)
        s1.metric("ROC-AUC (scale_pos_weight)", f"{roc_base:.3f}")
        s2.metric("Inválidos — SMOTE", f"{frac_inv_smote * 100:.1f}%")
        s3.metric("Inválidos — SMOTENC", f"{frac_inv_smotenc * 100:.1f}%")

        st.caption(
            "17 das 20 features são categóricas (13 binárias + 4 ordinais). O "
            "SMOTE interpola e gera valores fracionários inválidos em binárias; "
            "o SMOTENC usa a moda e mantém 0/1. Como o XGBoost já pondera a "
            "classe com `scale_pos_weight`, o oversampling não traz ganho — mas "
            "se for usado, deve ser SMOTENC."
        )

        sm_col1, sm_col2 = st.columns(2)
        with sm_col1:
            st.plotly_chart(fig_smote_integridade(integridade), use_container_width=True)
        with sm_col2:
            st.plotly_chart(fig_smote_comparacao(resultados), use_container_width=True)

# ==============================================================================
# ABA 4: SEGMENTAÇÃO POR CLUSTERS
# ==============================================================================
if aba == ABAS[3]:
    st.header("Segmentação de Perfis (K-Prototypes)")

    resultados = load_cluster_resultados()
    if resultados is None:
        st.info(
            "Resultados do clustering não encontrados em `data/resultados_cluster/`. "
            "Execute o script `src/clustering_comparacao.py` para gerar os dados desta aba."
        )
    else:
        dfc = resultados["dados"]
        resumo = resultados["resumo"]
        metricas = resultados["metricas"]
        comparacao = resultados["comparacao"]

        # k final = nº de clusters efetivamente usados (não o da silhueta)
        k_final = int(resumo["Cluster"].nunique())
        sil_k = metricas.loc[metricas["k"] == k_final, "silhueta"]
        sil_k = float(sil_k.iloc[0]) if len(sil_k) else float("nan")

        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        k_col1.metric("Clusters (k)", k_final)
        k_col2.metric("Registros Segmentados", f"{len(dfc):,}")
        k_col3.metric("Silhueta do k Escolhido", f"{sil_k:.3f}")
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
            st.plotly_chart(
                fig_cluster_cotovelo(metricas, k_escolhido=k_final),
                use_container_width=True,
            )
            st.caption(
                "A silhueta (calculada sobre contínuas padronizadas + one-hot das "
                "categóricas) favorece k=2, mas é ruidosa em dados mistos. O k final "
                f"(k={k_final}) foi escolhido pelo **gradiente de risco** + "
                "interpretabilidade — ver gráfico abaixo."
            )
            if comparacao is not None:
                st.plotly_chart(
                    fig_cluster_gradiente(comparacao, k_final),
                    use_container_width=True,
                )
                st.markdown("**Comparação de estratégias (subamostra de 20 mil):**")
                st.dataframe(
                    comparacao.round(3),
                    use_container_width=True,
                )
                st.caption(
                    "O HDBSCAN produz muitos clusters pequenos + ~22-26% de ruído "
                    "(não responde 'quais perfis existem?'); o GMM assume "
                    "gaussianidade violada pelas binárias. O K-Prototypes segue o "
                    "mais adequado para a base mista."
                )

# ==============================================================================
# ABA 5: SIMULADOR DE RISCO (MODELO SUPERVISIONADO)
# ==============================================================================
if aba == ABAS[4]:
    render_aba_simulador()

    # --- Explicabilidade global (SHAP) --------------------------------------
    st.markdown("---")
    st.subheader("🔎 Explicabilidade Global (SHAP)")

    shap_data = load_shap()
    if shap_data is None:
        st.info(
            "Resultados do SHAP global não encontrados em `data/modelo/`. "
            "Execute o script `src/shap_explain_pisi3.py`."
        )
    else:
        importancia = shap_data["importancia"]
        top_feature = importancia.iloc[0]

        h1, h2, h3 = st.columns(3)
        h1.metric("Feature mais influente", f"{top_feature['feature']}")
        h2.metric("Média |SHAP|", f"{top_feature['shap_media_absoluta']:.3f}")
        h3.metric("Registros analisados", f"{shap_data['shap'].shape[0]:,}")

        st.caption(
            "Contribuições SHAP (via `pred_contribs` nativo do XGBoost) sobre uma "
            "amostra estratificada do teste. À direita do zero, a feature aumenta "
            "o risco; à esquerda, reduz."
        )

        st.plotly_chart(
            fig_shap_importancia(importancia), use_container_width=True
        )
        st.plotly_chart(
            fig_shap_beeswarm(
                shap_data["shap"], shap_data["X"], shap_data["features"]
            ),
            use_container_width=True,
        )
