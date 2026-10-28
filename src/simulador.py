"""Aba Simulador de Risco do dashboard: predição individual de DCV.

Concentra a interface de teste em tempo real do modelo supervisionado
(artefato `data/modelo/dcv_modelo.pkl`, produzido por `src/train_model.py`
e consumido por `src/predictor.py`), mantendo o `app.py` responsável
apenas pelo layout geral.
"""

import streamlit as st

from src.dashboard_comps import fig_fatores_risco
from src.predictor import PacienteInvalido, PredictorDCV

# Perfis de exemplo (mesmos valores usados nos smoke tests do modelo)
PERFIS = {
    "Perfil neutro": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 27.0,
        "Smoker": 0, "Stroke": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
        "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0, "Sex": 1, "Age": 7, "Education": 4, "Income": 6,
    },
    "Perfil de alto risco": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.0,
        "Smoker": 1, "Stroke": 1, "PhysActivity": 0, "Fruits": 0,
        "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
        "NoDocbcCost": 0, "GenHlth": 5, "MentHlth": 5, "PhysHlth": 15,
        "DiffWalk": 1, "Sex": 1, "Age": 11, "Education": 3, "Income": 2,
    },
    "Perfil de baixo risco": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 23.0,
        "Smoker": 0, "Stroke": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
        "NoDocbcCost": 0, "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0, "Sex": 1, "Age": 3, "Education": 6, "Income": 8,
    },
}


@st.cache_resource
def load_predictor():
    """Carrega o artefato do modelo supervisionado; None se ainda não treinado."""
    try:
        return PredictorDCV.carregar()
    except FileNotFoundError:
        return None


def render_aba_simulador():
    st.header("Simulador de Risco — Predição Individual de DCV")

    pred = load_predictor()
    if pred is None:
        st.info(
            "Modelo não encontrado em `data/modelo/dcv_modelo.pkl`. "
            "Execute primeiro o script `src/train_model.py` para gerar o artefato desta aba."
        )
        return

    perfil = st.selectbox("Ponto de partida (predefine o formulário):", list(PERFIS))
    p0 = PERFIS[perfil]
    st.caption(
        "Ajuste os campos do paciente e clique em **Predizer risco** para "
        "executar o modelo XGBoost em tempo real sobre este perfil."
    )

    with st.form("form_paciente"):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown("**Condições clínicas**")
            highbp = st.toggle("Pressão alta diagnosticada", value=bool(p0["HighBP"]))
            highchol = st.toggle("Colesterol alto diagnosticado", value=bool(p0["HighChol"]))
            cholcheck = st.toggle("Colesterol checado (últimos 5 anos)", value=bool(p0["CholCheck"]))
            stroke = st.toggle("Já teve AVC/derrame", value=bool(p0["Stroke"]))
            diffwalk = st.toggle("Dificuldade para caminhar/subir degraus", value=bool(p0["DiffWalk"]))
            bmi = st.slider("IMC", min_value=12.0, max_value=60.0, value=float(p0["BMI"]), step=0.1)
            genhlth = st.slider("Saúde geral percebida (1=excelente a 5=péssima)", 1, 5, int(p0["GenHlth"]))
            physhlth = st.slider("Dias com saúde física ruim (últimos 30)", 0, 30, int(p0["PhysHlth"]))
            menthlth = st.slider("Dias com saúde mental ruim (últimos 30)", 0, 30, int(p0["MentHlth"]))

        with f_col2:
            st.markdown("**Hábitos e acesso à saúde**")
            smoker = st.toggle("Fumou 100+ cigarros na vida", value=bool(p0["Smoker"]))
            hvyalcohol = st.toggle("Consumo pesado de álcool", value=bool(p0["HvyAlcoholConsump"]))
            physactivity = st.toggle("Atividade física nos últimos 30 dias", value=bool(p0["PhysActivity"]))
            fruits = st.toggle("Consome frutas todos os dias", value=bool(p0["Fruits"]))
            veggies = st.toggle("Consome vegetais todos os dias", value=bool(p0["Veggies"]))
            anyhealthcare = st.toggle("Possui plano/acesso à saúde", value=bool(p0["AnyHealthcare"]))
            nodocbccost = st.toggle("Deixou de consultar por custo (12 meses)", value=bool(p0["NoDocbcCost"]))

        with f_col3:
            st.markdown("**Perfil sociodemográfico**")
            sexo_masc = st.toggle("Sexo masculino", value=bool(p0["Sex"]))
            idade = st.slider("Faixa de idade (1=18-24 a 13=80+)", 1, 13, int(p0["Age"]))
            educacao = st.slider("Escolaridade (1=sem estudo a 6=superior+)", 1, 6, int(p0["Education"]))
            renda = st.slider("Renda (1=menor a 8=maior, escala BRFSS)", 1, 8, int(p0["Income"]))

        enviado = st.form_submit_button("Predizer risco", type="primary")

    if not enviado:
        st.info("Ajuste os campos e clique em **Predizer risco** para rodar o modelo neste perfil.")
        return

    paciente = {
        "HighBP": int(highbp), "HighChol": int(highchol),
        "CholCheck": int(cholcheck), "BMI": float(bmi),
        "Smoker": int(smoker), "Stroke": int(stroke),
        "PhysActivity": int(physactivity), "Fruits": int(fruits),
        "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcohol),
        "AnyHealthcare": int(anyhealthcare), "NoDocbcCost": int(nodocbccost),
        "GenHlth": int(genhlth), "MentHlth": int(menthlth),
        "PhysHlth": int(physhlth), "DiffWalk": int(diffwalk),
        "Sex": int(sexo_masc), "Age": int(idade),
        "Education": int(educacao), "Income": int(renda),
    }
    try:
        resultado = pred.prever(paciente)
    except PacienteInvalido as erro:
        st.error(f"Erro de contrato: {erro}")
        return

    prob = resultado["probabilidade"]
    classe = resultado["classificacao"]
    emoji = {"alto": "🔴", "moderado": "🟠", "baixo": "🟢"}[classe]

    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:
        st.metric("Probabilidade estimada", f"{prob:.1%}")
        st.markdown(f"### {emoji} Risco **{classe}**")
        st.progress(min(prob, 1.0))
        st.caption(
            f"Threshold de triagem: {pred.threshold:.2f} "
            f"(risco 'alto' a partir de {pred.threshold:.0%}; "
            f"'moderado' entre {pred.threshold / 2:.0%} e {pred.threshold:.0%})."
        )
    with r_col2:
        st.plotly_chart(
            fig_fatores_risco(resultado["fatores_principais"]),
            use_container_width=True,
        )

    met = pred.metricas_teste
    st.markdown("---")
    st.caption(
        "⚠️ Ferramenta de **triagem educacional**, não um diagnóstico: o modelo foi "
        "treinado no BRFSS 2015 (inquérito transversal e autorrelatado) e estima a "
        "probabilidade de o perfil corresponder a alguém **com** doença "
        "cardiovascular. Como a base é desbalanceada (~10% de casos) e o modelo usa "
        "peso de classe, o valor bruto serve mais para ordenar perfis do que como "
        "frequência real. "
        f"Referência de qualidade (hold-out de 20%): recall {met['recall']:.1%} | "
        f"precisão {met['precisao']:.1%} | PR-AUC {met['pr_auc']:.3f}."
    )

