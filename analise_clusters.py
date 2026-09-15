#!/usr/bin/env python3
"""
Segmentação de Perfis de Saúde via K-Means.

Usa a saída de pre_processamento.py (features padronizadas) para agrupar os
registros em clusters de perfil de risco cardiovascular, escolhe o número de
clusters via método do cotovelo + silhouette, e cruza cada cluster com a
prevalência real de doença cardíaca (HeartDiseaseorAttack) e com os valores
originais (não escalados) para dar interpretabilidade ao resultado.

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

FEATURES_CLUSTERING = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "HvyAlcoholConsump",
    "PhysActivity", "Age", "Sex", "Stroke", "GenHlth", "DiffWalk",
    "Diabetes_012_0.0", "Diabetes_012_1.0", "Diabetes_012_2.0",
]

ROTULOS_AGE = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+",
}
ROTULOS_GENHLTH = {1: "Excelente", 2: "Muito boa", 3: "Boa", 4: "Razoável", 5: "Ruim"}
ROTULOS_DIABETES = {0: "Sem diabetes", 1: "Pré-diabetes", 2: "Diabetes"}


def escolher_k(X: np.ndarray, k_min: int = 2, k_max: int = 7) -> tuple[list[dict], int]:
    resultados = []
    amostra_idx = np.random.RandomState(SEMENTE).choice(len(X), size=min(6000, len(X)), replace=False)
    for k in range(k_min, k_max + 1):
        modelo = MiniBatchKMeans(n_clusters=k, random_state=SEMENTE, n_init=10, batch_size=2048)
        labels = modelo.fit_predict(X)
        silhueta = silhouette_score(X[amostra_idx], labels[amostra_idx])
        resultados.append({"k": k, "inercia": float(modelo.inertia_), "silhueta": float(silhueta)})
        print(f"k={k}: inércia={modelo.inertia_:,.0f}  silhueta={silhueta:.4f}")
    melhor = max(resultados, key=lambda r: (round(r["silhueta"], 2), -r["k"]))
    return resultados, melhor["k"]


def main() -> None:
    pre_processado = pd.read_csv(PASTA_DADOS / "diabetes_cardiaco_pre_processado.csv")
    tratado_original = pd.read_csv(PASTA_DADOS / "diabetes_cardiaco_tratado.csv")
    assert len(pre_processado) == len(tratado_original), "Linhas desalinhadas entre os dois CSVs."

    features_presentes = [f for f in FEATURES_CLUSTERING if f in pre_processado.columns]
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
            "prevalencia_doenca_cardiaca_pct": round(100 * grupo["HeartDiseaseorAttack"].mean(), 2),
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
            "distribuicao_diabetes": {ROTULOS_DIABETES.get(int(k), str(k)): round(100 * v, 1) for k, v in diabetes_dist.items()},
        })

    prevalencia_geral = round(100 * base["HeartDiseaseorAttack"].mean(), 2)

    partes = []
    for c in sorted(base["cluster"].unique()):
        grupo = base[base["cluster"] == c]
        partes.append(grupo.sample(n=min(600, len(grupo)), random_state=SEMENTE))
    amostra_scatter = pd.concat(partes, ignore_index=True)

    pontos = [
        {"x": round(float(x), 3), "y": round(float(y), 3), "cluster": int(c), "doenca": int(d)}
        for x, y, c, d in zip(
            amostra_scatter["pca_x"], amostra_scatter["pca_y"],
            amostra_scatter["cluster"], amostra_scatter["HeartDiseaseorAttack"],
        )
    ]

    saida = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "total_registros": total,
        "features_usadas": features_presentes,
        "prevalencia_geral_doenca_cardiaca_pct": prevalencia_geral,
        "varredura_k": varredura_k,
        "k_escolhido": k_escolhido,
        "variancia_explicada_pca": [round(v, 4) for v in variancia_explicada],
        "perfis": sorted(perfis, key=lambda p: p["prevalencia_doenca_cardiaca_pct"]),
        "pontos_dispersao": pontos,
    }

    with (PASTA_DADOS / "dados_clusters.json").open("w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False)

    print(f"\nOK: {len(pontos)} pontos de amostra, {len(perfis)} clusters.")
    print(f"Salvo em {PASTA_DADOS / 'dados_clusters.json'}")


if __name__ == "__main__":
    main()
