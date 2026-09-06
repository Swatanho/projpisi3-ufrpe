#!/usr/bin/env python3
"""
Dependências:
    pip install pandas scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


ALVO = "HeartDiseaseorAttack"
ATRIBUTOS_RELEVANTES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Diabetes_012", "Smoker",
    "HvyAlcoholConsump", "PhysActivity", "Age", "Sex", "Stroke",
    "GenHlth", "DiffWalk",
]

# Carregar o Arquivo

def ler_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, low_memory=False)

# Escolhe os atributos necessários

def selecionar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente o alvo e os atributos definidos pela equipe."""
    colunas = [ALVO, *ATRIBUTOS_RELEVANTES]
    ausentes = [coluna for coluna in colunas if coluna not in df.columns]
    if ausentes:
        dica = ""
        if "Diabetes_012" in ausentes and "Diabetes_binary" in df.columns:
            dica = (
                " Use diabetes_012_health_indicators_BRFSS2015.csv, "
                "não uma das versões binárias."
            )
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(ausentes) + "." + dica
        )

    selecionado = df.loc[:, colunas].copy()
    for coluna in colunas:
        selecionado[coluna] = pd.to_numeric(selecionado[coluna], errors="coerce")

    faltantes = selecionado.isna().sum()
    faltantes = faltantes[faltantes > 0]
    if not faltantes.empty:
        detalhes = ", ".join(f"{c}={int(n)}" for c, n in faltantes.items())
        raise ValueError(
            "Há valores ausentes ou não numéricos. Corrija-os antes: " + detalhes
        )
    return selecionado

# Remoção de redundância

def remover_sem_variancia(
    df: pd.DataFrame, limite_variancia: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    if limite_variancia < 0:
        raise ValueError("O limite de variância não pode ser negativo.")

    preditores = df.drop(columns=ALVO)
    seletor = VarianceThreshold(threshold=limite_variancia)
    seletor.fit(preditores)
    mantidas = preditores.columns[seletor.get_support()].tolist()
    removidas = [c for c in preditores.columns if c not in mantidas]
    return df.loc[:, [ALVO, *mantidas]].copy(), removidas


# Aplicação do Isolation Forest

def tratar_outliers(
    df: pd.DataFrame,
    contaminacao: str | float = "auto",
    semente: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preditores = df.drop(columns=ALVO)
    if preditores.shape[1] == 0:
        raise ValueError("Nenhum atributo restou para detectar outliers.")

    # O alvo não entra no modelo, evitando vazamento de informação.
    dados_escalados = StandardScaler().fit_transform(preditores)
    modelo = IsolationForest(
        n_estimators=200,
        contamination=contaminacao,
        random_state=semente,
        n_jobs=-1,
    )
    classificacao = modelo.fit_predict(dados_escalados)
    pontuacao = modelo.decision_function(dados_escalados)
    mascara_outlier = classificacao == -1

    outliers = df.loc[mascara_outlier].copy()
    outliers.insert(0, "indice_original", outliers.index)
    outliers["pontuacao_isolation_forest"] = pontuacao[mascara_outlier]
    tratados = df.loc[~mascara_outlier].copy().reset_index(drop=True)
    return tratados, outliers.reset_index(drop=True)

# SEÇÃO DE COMPARAÇÃO DE DADOS

def resumo_colunas(df: pd.DataFrame, etapa: str) -> pd.DataFrame:
    resumo = df.describe(include="all").T.reset_index(names="coluna")
    resumo.insert(0, "etapa", etapa)
    resumo["valores_ausentes"] = [int(df[c].isna().sum()) for c in resumo["coluna"]]
    resumo["valores_unicos"] = [int(df[c].nunique(dropna=False)) for c in resumo["coluna"]]
    return resumo


def gerar_comparacao(
    antes: pd.DataFrame,
    depois: pd.DataFrame,
    removidas: list[str],
    total_outliers: int,
    pasta_saida: Path,
) -> None:
    percentual = (total_outliers / len(antes) * 100) if len(antes) else 0.0
    comparacao = {
        "antes": {"linhas": len(antes), "colunas": antes.shape[1]},
        "depois": {"linhas": len(depois), "colunas": depois.shape[1]},
        "linhas_removidas_como_outliers": total_outliers,
        "percentual_de_linhas_removidas": round(percentual, 4),
        "atributos_removidos_por_variancia": removidas,
        "distribuicao_alvo_antes": {
            str(k): int(v) for k, v in antes[ALVO].value_counts().sort_index().items()
        },
        "distribuicao_alvo_depois": {
            str(k): int(v) for k, v in depois[ALVO].value_counts().sort_index().items()
        },
    }
    (pasta_saida / "comparacao_antes_depois.json").write_text(
        json.dumps(comparacao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(
        [resumo_colunas(antes, "antes"), resumo_colunas(depois, "depois")],
        ignore_index=True,
    ).to_csv(pasta_saida / "estatisticas_antes_depois.csv", index=False)

####

def executar_tratamento(
    entrada: Path,
    pasta_saida: Path,
    limite_variancia: float,
    contaminacao: str | float,
) -> None:
    original = ler_dataset(entrada)
    selecionado = selecionar_atributos(original)
    sem_redundancia, removidas = remover_sem_variancia(
        selecionado, limite_variancia
    )
    tratado, outliers = tratar_outliers(sem_redundancia, contaminacao)

    pasta_saida.mkdir(parents=True, exist_ok=True)
    tratado.to_csv(pasta_saida / "diabetes_cardiaco_tratado.csv", index=False)
    outliers.to_csv(pasta_saida / "registros_outliers.csv", index=False)
    gerar_comparacao(
        selecionado, tratado, removidas, len(outliers), pasta_saida
    )

    print("Tratamento concluído.")
    print(f"Linhas antes: {len(selecionado):,}")
    print(f"Linhas depois: {len(tratado):,}")
    print(f"Outliers removidos: {len(outliers):,}")
    print(f"Atributos removidos por variância: {removidas or 'nenhum'}")
    print(f"Resultados: {pasta_saida.resolve()}")


def interpretar_contaminacao(valor: str) -> str | float:
    """Aceita 'auto' ou uma proporção entre 0 e 0,5."""
    if valor.lower() == "auto":
        return "auto"
    try:
        numero = float(valor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use 'auto' ou um número maior que 0 e menor ou igual a 0.5."
        ) from exc
    if not 0 < numero <= 0.5:
        raise argparse.ArgumentTypeError(
            "A contaminação deve ser maior que 0 e menor ou igual a 0.5."
        )
    return numero


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
    "entrada",
    type=Path,
    nargs="?",
    default=Path("DB/diabetes_012_health_indicators_BRFSS2015.csv"),
    help="Caminho do CSV original",
)
    parser.add_argument(
        "--saida-dir", type=Path, default=Path("DB/dados_tratados"),
        help="Pasta de saída (padrão: DB/dados_tratados)",
    )
    parser.add_argument(
        "--limite-variancia", type=float, default=0.0,
        help="Remove atributos com variância até este valor (padrão: 0)",
    )
    parser.add_argument(
        "--contaminacao", type=interpretar_contaminacao, default="auto",
        help="Proporção esperada de outliers ou 'auto' (padrão: auto)",
    )
    return parser.parse_args()


def main() -> int:
    args = argumentos()
    try:
        executar_tratamento(
            args.entrada, args.saida_dir, args.limite_variancia, args.contaminacao
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
