#!/usr/bin/env python3
"""
Segmentação de Perfis de Saúde via K-Means.

Usa a saída de pre_processamento.py (features escaladas) para agrupar os
registros em perfis de saúde, escolhe o número de clusters por silhouette +
inércia e cruza cada cluster com a ocorrência observada de
HeartDiseaseorAttack para interpretação.

Também gera estatísticas globais da base e associações descritivas entre
indicadores de saúde e o desfecho. Essas associações NÃO representam causas.

Gera DB/dados_tratados/dados_clusters.json, consumido pelo painel HTML.

Dependências:
    pip install pandas scikit-learn
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SEMENTE = 42
PASTA_DADOS = Path("DB/dados_tratados")
ALVO = "HeartDiseaseorAttack"

FEATURES_CLUSTERING = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "HvyAlcoholConsump",
    "PhysActivity", "Age", "Sex", "Stroke", "GenHlth", "DiffWalk",
]

ROTULOS_AGE = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+",
}
ROTULOS_GENHLTH = {1: "Excelente", 2: "Muito boa", 3: "Boa", 4: "Razoável", 5: "Ruim"}
ROTULOS_DIABETES = {0: "Sem diabetes", 1: "Pré-diabetes", 2: "Diabetes"}
ROTULOS_SEX = {0: "Feminino", 1: "Masculino"}

DESCRICOES_FEATURES = {
    "HighBP": "Pressão arterial alta",
    "HighChol": "Colesterol alto",
    "CholCheck": "Verificação de colesterol",
    "BMI": "Índice de Massa Corporal (IMC)",
    "Diabetes_012": "Classificação de diabetes",
    "Smoker": "Tabagismo",
    "HvyAlcoholConsump": "Consumo elevado de álcool",
    "PhysActivity": "Atividade física",
    "Age": "Faixa etária",
    "Sex": "Sexo",
    "Stroke": "Histórico de AVC",
    "GenHlth": "Saúde geral percebida",
    "DiffWalk": "Dificuldade para caminhar",
}


def escolher_k(X: np.ndarray, k_min: int = 2, k_max: int = 7) -> tuple[list[dict], int]:
    resultados = []
    amostra_idx = np.random.RandomState(SEMENTE).choice(
        len(X), size=min(6000, len(X)), replace=False
    )
    for k in range(k_min, k_max + 1):
        modelo = MiniBatchKMeans(
            n_clusters=k, random_state=SEMENTE, n_init=10, batch_size=2048
        )
        labels = modelo.fit_predict(X)
        silhueta = silhouette_score(X[amostra_idx], labels[amostra_idx])
        resultados.append({"k": k, "inercia": float(modelo.inertia_), "silhueta": float(silhueta)})
        print(f"k={k}: inércia={modelo.inertia_:,.0f}  silhueta={silhueta:.4f}")
    melhor = max(resultados, key=lambda r: (round(r["silhueta"], 2), -r["k"]))
    return resultados, melhor["k"]


def distribuicoes_base(base: pd.DataFrame) -> dict:
    """Monta estatísticas globais para detalhar a base no painel."""
    def dist_percentual(coluna: str, rotulos: dict | None = None) -> list[dict]:
        serie = base[coluna].value_counts(normalize=True).sort_index()
        resultado = []
        for chave, valor in serie.items():
            numero = int(chave) if pd.notna(chave) else chave
            resultado.append({
                "codigo": numero,
                "categoria": (rotulos or {}).get(numero, str(numero)),
                "percentual": round(float(valor * 100), 2),
            })
        return resultado

    binarias = [
        "HighBP", "HighChol", "CholCheck", "Smoker",
        "HvyAlcoholConsump", "PhysActivity", "Stroke", "DiffWalk",
    ]
    indicadores = []
    for coluna in binarias:
        percentual = 100 * float(base[coluna].mean())
        indicadores.append({
            "variavel": coluna,
            "nome": DESCRICOES_FEATURES[coluna],
            "percentual_1": round(percentual, 2),
            "percentual_0": round(100 - percentual, 2),
        })

    return {
        "diabetes": dist_percentual("Diabetes_012", ROTULOS_DIABETES),
        "saude_geral": dist_percentual("GenHlth", ROTULOS_GENHLTH),
        "faixa_etaria": dist_percentual("Age", ROTULOS_AGE),
        "sexo": dist_percentual("Sex", ROTULOS_SEX),
        "indicadores_binarios": indicadores,
        "bmi": {
            "media": round(float(base["BMI"].mean()), 2),
            "mediana": round(float(base["BMI"].median()), 2),
            "min": round(float(base["BMI"].min()), 2),
            "max": round(float(base["BMI"].max()), 2),
        },
    }


def associacoes_descritivas(base: pd.DataFrame) -> list[dict]:
    """
    Compara a prevalência observada do desfecho quando um indicador binário
    está presente/ausente. É uma comparação descritiva, não causal.
    """
    colunas = [
        "HighBP", "HighChol", "CholCheck", "Smoker",
        "HvyAlcoholConsump", "PhysActivity", "Stroke", "DiffWalk",
    ]
    registros = []
    for coluna in colunas:
        com = base.loc[base[coluna] == 1, ALVO]
        sem = base.loc[base[coluna] == 0, ALVO]
        if len(com) == 0 or len(sem) == 0:
            continue
        taxa_com = 100 * float(com.mean())
        taxa_sem = 100 * float(sem.mean())
        registros.append({
            "variavel": coluna,
            "nome": DESCRICOES_FEATURES[coluna],
            "prevalencia_com_indicador_pct": round(taxa_com, 2),
            "prevalencia_sem_indicador_pct": round(taxa_sem, 2),
            "diferenca_pontos_percentuais": round(taxa_com - taxa_sem, 2),
            "n_com_indicador": int(len(com)),
            "n_sem_indicador": int(len(sem)),
        })
    return sorted(
        registros,
        key=lambda x: abs(x["diferenca_pontos_percentuais"]),
        reverse=True,
    )


def perfil_interpretativo(base: pd.DataFrame) -> list[dict]:
    perfis = []
    total = len(base)
    for c in sorted(base["cluster"].unique()):
        grupo = base[base["cluster"] == c]
        n = len(grupo)
        diabetes_dist = grupo["Diabetes_012"].value_counts(normalize=True).sort_index()
        genhlth_dist = grupo["GenHlth"].value_counts(normalize=True).sort_index()
        age_dist = grupo["Age"].value_counts(normalize=True).sort_index()

        perfis.append({
            "cluster": int(c),
            "tamanho": int(n),
            "percentual_do_total": round(100 * n / total, 2),
            "prevalencia_doenca_cardiaca_pct": round(100 * grupo[ALVO].mean(), 2),
            "bmi_medio": round(float(grupo["BMI"].mean()), 1),
            "pct_pressao_alta": round(100 * grupo["HighBP"].mean(), 1),
            "pct_colesterol_alto": round(100 * grupo["HighChol"].mean(), 1),
            "pct_fumante": round(100 * grupo["Smoker"].mean(), 1),
            "pct_dificuldade_caminhar": round(100 * grupo["DiffWalk"].mean(), 1),
            "pct_atividade_fisica": round(100 * grupo["PhysActivity"].mean(), 1),
            "pct_consumo_alcool_pesado": round(100 * grupo["HvyAlcoholConsump"].mean(), 1),
            "pct_masculino": round(100 * grupo["Sex"].mean(), 1),
            "faixa_etaria_predominante": ROTULOS_AGE.get(int(age_dist.idxmax()), "?"),
            "saude_geral_predominante": ROTULOS_GENHLTH.get(int(genhlth_dist.idxmax()), "?"),
            "diabetes_predominante": ROTULOS_DIABETES.get(int(diabetes_dist.idxmax()), "?"),
            "distribuicao_diabetes": {
                ROTULOS_DIABETES.get(int(k), str(k)): round(100 * v, 1)
                for k, v in diabetes_dist.items()
            },
        })
    return perfis


def main() -> None:
    pre_processado = pd.read_csv(PASTA_DADOS / "diabetes_cardiaco_pre_processado.csv")
    tratado_original = pd.read_csv(PASTA_DADOS / "diabetes_cardiaco_tratado.csv")
    assert len(pre_processado) == len(tratado_original), "Linhas desalinhadas entre os dois CSVs."

    features_presentes = [f for f in FEATURES_CLUSTERING if f in pre_processado.columns]
    # O OneHotEncoder pode produzir Diabetes_012_0, Diabetes_012_1, ...
    # ou Diabetes_012_0.0, dependendo da versão/tipos do sklearn/pandas.
    diabetes_ohe = sorted(c for c in pre_processado.columns if c.startswith("Diabetes_012_"))
    if not diabetes_ohe:
        print("Aviso: nenhuma coluna OHE de Diabetes_012 foi encontrada.")
    features_presentes.extend(c for c in diabetes_ohe if c not in features_presentes)
    ausentes = [f for f in FEATURES_CLUSTERING if f not in pre_processado.columns]
    if ausentes:
        print(f"Aviso: features não encontradas e ignoradas: {ausentes}")
    X = pre_processado[features_presentes].to_numpy()

    print("Buscando o número de clusters (k)...")
    varredura_k, k_escolhido = escolher_k(X)
    print(f"\nk escolhido: {k_escolhido}")

    modelo_final = KMeans(n_clusters=k_escolhido, random_state=SEMENTE, n_init=20)
    labels = modelo_final.fit_predict(X)

    pca = PCA(n_components=2, random_state=SEMENTE)
    coords = pca.fit_transform(X)
    variancia_explicada = pca.explained_variance_ratio_.tolist()

    base = tratado_original.copy()
    base["cluster"] = labels
    base["pca_x"] = coords[:, 0]
    base["pca_y"] = coords[:, 1]

    perfis = perfil_interpretativo(base)
    prevalencia_geral = round(100 * base[ALVO].mean(), 2)

    partes = []
    for c in sorted(base["cluster"].unique()):
        grupo = base[base["cluster"] == c]
        partes.append(grupo.sample(n=min(600, len(grupo)), random_state=SEMENTE))
    amostra_scatter = pd.concat(partes, ignore_index=True)

    pontos = [
        {"x": round(float(x), 3), "y": round(float(y), 3), "cluster": int(c), "doenca": int(d)}
        for x, y, c, d in zip(
            amostra_scatter["pca_x"], amostra_scatter["pca_y"],
            amostra_scatter["cluster"], amostra_scatter[ALVO],
        )
    ]

    saida = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "fonte": {
            "dataset": "Diabetes Health Indicators Dataset",
            "arquivo": "diabetes_012_health_indicators_BRFSS2015.csv",
            "origem": "Kaggle / BRFSS 2015",
            "url": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
            "observacao": "Base observacional: relações encontradas são associações descritivas e não demonstram causalidade.",
        },
        "total_registros": total if (total := len(base)) else 0,
        "total_colunas_tratadas": int(tratado_original.shape[1]),
        "features_usadas": features_presentes,
        "prevalencia_geral_doenca_cardiaca_pct": prevalencia_geral,
        "varredura_k": varredura_k,
        "k_escolhido": k_escolhido,
        "variancia_explicada_pca": [round(v, 4) for v in variancia_explicada],
        "resumo_base": distribuicoes_base(tratado_original),
        "associacoes_descritivas": associacoes_descritivas(tratado_original),
        "perfis": sorted(perfis, key=lambda p: p["prevalencia_doenca_cardiaca_pct"]),
        "pontos_dispersao": pontos,
        "interpretacao": {
            "objetivo": "Encontrar grupos de respondentes com combinações semelhantes de indicadores clínicos, demográficos, funcionais e comportamentais.",
            "motivo_kmeans": "K-Means permite formar grupos por similaridade entre as características após o escalonamento.",
            "motivo_pca": "PCA reduz as 15 dimensões usadas no clustering para duas componentes apenas para visualização.",
            "motivo_validacao": "A variável-alvo HeartDiseaseorAttack não participa da formação dos clusters; ela é usada posteriormente para verificar se os perfis apresentam prevalências observadas distintas.",
        },
    }

    with (PASTA_DADOS / "dados_clusters.json").open("w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)

    print(f"\nOK: {len(pontos)} pontos de amostra, {len(perfis)} clusters.")
    print(f"Salvo em {PASTA_DADOS / 'dados_clusters.json'}")


if __name__ == "__main__":
    main()
