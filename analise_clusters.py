"""
Gera as três análises exploratórias do painel.

1. Gênero vs. fatores de risco e evento cardíaco.
2. Comportamento preventivo vs. barreira financeira.
3. Determinantes sociais (renda e educação) vs. saúde percebida e diabetes.

A etapa é descritiva. Os cruzamentos não demonstram causalidade.

Dependências:
    pip install pandas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ALVO = "HeartDiseaseorAttack"
SEXOS = {0: "Mulher", 1: "Homem"}
RENDA_LABELS = {i: f"Renda {i}" for i in range(1, 9)}
EDUCACAO_LABELS = {i: f"Escolaridade {i}" for i in range(1, 7)}

REQUERIDAS = [
    ALVO, "Sex", "Smoker", "HvyAlcoholConsump", "PhysActivity",
    "CholCheck", "AnyHealthcare", "NoDocbcCost",
    "Income", "Education", "GenHlth", "Diabetes_012",
]


def carregar(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    faltantes = [c for c in REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError("Colunas necessárias ausentes: " + ", ".join(faltantes))
    for c in REQUERIDAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[REQUERIDAS].isna().any().any():
        falt = df[REQUERIDAS].isna().sum()
        falt = falt[falt > 0]
        raise ValueError("Há valores ausentes/não numéricos nas variáveis da análise: " + ", ".join(f"{c}={int(n)}" for c, n in falt.items()))
    # Diabetes_binary: 1 = pré-diabetes ou diabetes; 0 = sem diabetes.
    df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
    df["Sedentario"] = (df["PhysActivity"] == 0).astype(int)
    return df


def percentual_por_sexo(df: pd.DataFrame, coluna: str) -> list[dict]:
    out = []
    for codigo, rotulo in SEXOS.items():
        grupo = df[df["Sex"] == codigo]
        out.append({"sex": codigo, "sexo": rotulo, "percentual": round(100 * float(grupo[coluna].mean()), 2), "n": int(len(grupo))})
    return out


def grafico1(df: pd.DataFrame) -> dict:
    metricas = [
        (ALVO, "Doença cardíaca / ataque cardíaco", "desfecho"),
        ("Smoker", "Tabagismo", "risco comportamental"),
        ("HvyAlcoholConsump", "Consumo elevado de álcool", "risco comportamental"),
        ("Sedentario", "Sem atividade física", "comportamento associado"),
    ]
    dados = []
    for coluna, nome, papel in metricas:
        por_sexo = percentual_por_sexo(df, coluna)
        dados.append({"variavel": coluna, "nome": nome, "papel": papel, "valores": por_sexo})
    homem = {d["variavel"]: next(v["percentual"] for v in d["valores"] if v["sex"] == 1) for d in dados}
    mulher = {d["variavel"]: next(v["percentual"] for v in d["valores"] if v["sex"] == 0) for d in dados}
    return {
        "titulo": "Gênero vs. fatores de risco e evento cardíaco",
        "dados": dados,
        "resumo": {
            "taxa_cardiaca_mulher": mulher[ALVO],
            "taxa_cardiaca_homem": homem[ALVO],
            "diferenca_cardiaca_homem_menos_mulher_pp": round(homem[ALVO] - mulher[ALVO], 2),
            "tabagismo_homem": homem["Smoker"],
            "tabagismo_mulher": mulher["Smoker"],
            "alcool_homem": homem["HvyAlcoholConsump"],
            "alcool_mulher": mulher["HvyAlcoholConsump"],
            "sedentarismo_homem": homem["Sedentario"],
            "sedentarismo_mulher": mulher["Sedentario"],
        },
    }


def grafico2(df: pd.DataFrame) -> dict:
    metricas = [
        ("CholCheck", "Fez check-up de colesterol", "prevenção"),
        ("AnyHealthcare", "Tem cobertura de saúde", "acesso"),
        ("NoDocbcCost", "Deixou de ir ao médico por custo", "barreira financeira"),
    ]
    dados = []
    for coluna, nome, papel in metricas:
        dados.append({"variavel": coluna, "nome": nome, "papel": papel, "valores": percentual_por_sexo(df, coluna)})

    def val(col, sex):
        return round(100 * float(df.loc[df["Sex"] == sex, col].mean()), 2)

    return {
        "titulo": "Comportamento preventivo e barreira financeira",
        "dados": dados,
        "resumo": {
            "cholcheck_mulher": val("CholCheck", 0),
            "cholcheck_homem": val("CholCheck", 1),
            "cobertura_mulher": val("AnyHealthcare", 0),
            "cobertura_homem": val("AnyHealthcare", 1),
            "barreira_custo_mulher": val("NoDocbcCost", 0),
            "barreira_custo_homem": val("NoDocbcCost", 1),
        },
    }


def matriz_heatmap(df: pd.DataFrame, valor_coluna: str, agregacao: str) -> dict:
    if agregacao == "media":
        tabela = df.pivot_table(index="Education", columns="Income", values=valor_coluna, aggfunc="mean")
    elif agregacao == "taxa":
        tabela = df.pivot_table(index="Education", columns="Income", values=valor_coluna, aggfunc="mean") * 100
    else:
        raise ValueError("Agregação inválida")

    tabela = tabela.reindex(index=range(1, 7), columns=range(1, 9))
    valores = []
    for education in range(1, 7):
        linha = []
        for income in range(1, 9):
            valor = tabela.loc[education, income]
            linha.append(None if pd.isna(valor) else round(float(valor), 2))
        valores.append(linha)
    return {
        "linhas": [{"codigo": i, "label": EDUCACAO_LABELS[i]} for i in range(1, 7)],
        "colunas": [{"codigo": i, "label": RENDA_LABELS[i]} for i in range(1, 9)],
        "valores": valores,
    }


def grafico3(df: pd.DataFrame) -> dict:
    gen_hlth_income = df.groupby("Income")["GenHlth"].mean().reindex(range(1, 9))
    diab_income = df.groupby("Income")["Diabetes_binary"].mean().reindex(range(1, 9)) * 100
    gen_hlth_edu = df.groupby("Education")["GenHlth"].mean().reindex(range(1, 7))
    diab_edu = df.groupby("Education")["Diabetes_binary"].mean().reindex(range(1, 7)) * 100
    return {
        "titulo": "Determinantes sociais da saúde",
        "heatmap_genhlth": matriz_heatmap(df, "GenHlth", "media"),
        "heatmap_diabetes": matriz_heatmap(df, "Diabetes_binary", "taxa"),
        "resumo": {
            "genhlth_renda_menor": round(float(gen_hlth_income.iloc[0]), 2),
            "genhlth_renda_maior": round(float(gen_hlth_income.iloc[-1]), 2),
            "diabetes_renda_menor_pct": round(float(diab_income.iloc[0]), 2),
            "diabetes_renda_maior_pct": round(float(diab_income.iloc[-1]), 2),
            "genhlth_educacao_menor": round(float(gen_hlth_edu.iloc[0]), 2),
            "genhlth_educacao_maior": round(float(gen_hlth_edu.iloc[-1]), 2),
            "diabetes_educacao_menor_pct": round(float(diab_edu.iloc[0]), 2),
            "diabetes_educacao_maior_pct": round(float(diab_edu.iloc[-1]), 2),
        },
        "nota": "GenHlth é ordinal: valores menores representam melhor saúde percebida. Diabetes_binary é derivada de Diabetes_012.",
    }


def conclusoes(df: pd.DataFrame, g1: dict, g2: dict, g3: dict) -> dict:
    r1, r2, r3 = g1["resumo"], g2["resumo"], g3["resumo"]
    sexo_cardiaco = "homens" if r1["diferenca_cardiaca_homem_menos_mulher_pp"] > 0 else "mulheres" if r1["diferenca_cardiaca_homem_menos_mulher_pp"] < 0 else "homens e mulheres"
    return {
        "grafico1": f"Na amostra, {sexo_cardiaco} apresentam a maior taxa observada de doença cardíaca/ataque cardíaco. A comparação deve ser lida junto com tabagismo, álcool e ausência de atividade física; ela descreve associação e não permite atribuir o resultado a um único fator.",
        "grafico2": f"O gráfico permite comparar diretamente o percentual de mulheres e homens que fizeram verificação de colesterol, possuem cobertura de saúde e relataram não ir ao médico por causa do custo. Esses números ajudam a separar comportamento preventivo de barreira financeira sem assumir que a diferença decorra exclusivamente de escolha individual.",
        "grafico3": f"A combinação de renda e escolaridade mostra como o perfil socioeconômico se distribui junto com a saúde percebida e o diabetes. Como GenHlth vai de 1 (melhor) a 5 (pior), valores maiores significam pior saúde percebida. A comparação entre as faixas mais baixa e mais alta é descritiva e não prova que renda ou educação causem o desfecho.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("DB/dados_tratados/diabetes_cardiaco_tratado.csv"))
    parser.add_argument("--saida-dir", type=Path, default=Path("DB/dados_tratados"))
    args = parser.parse_args()

    df = carregar(args.entrada)
    g1 = grafico1(df)
    g2 = grafico2(df)
    g3 = grafico3(df)
    saida = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "fonte": {
            "dataset": "Diabetes Health Indicators Dataset",
            "arquivo": args.entrada.name,
            "origem": "Kaggle / BRFSS 2015",
            "url": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
        },
        "total_registros": int(len(df)),
        "graficos": {"genero_riscos": g1, "prevencao_barreira": g2, "socioeconomico": g3},
        "conclusoes": conclusoes(df, g1, g2, g3),
        "observacao_metodologica": "As comparações são descritivas. Diferenças entre grupos não demonstram causalidade.",
    }
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    caminho = args.saida_dir / "dados_graficos.json"
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Análises geradas em {caminho.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
