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
from sklearn.feature_selection import VarianceThreshold


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


# Tratamento de valores extremos do BMI (WINSORIZAÇÃO de valores extremos de BMI)
#
# O BMI é a única variável contínua do conjunto. As demais colunas são fatores
# de risco binários/ordinais, cujas combinações raras representam justamente os
# grupos de maior risco e não devem ser descartadas. Por isso, em vez de remover
# linhas, aplicamos winsorização simétrica: valores acima do percentil superior
# são limitados ao teto e valores abaixo do percentil inferior são limitados ao
# piso, preservando todos os registros e a distribuição da classe alvo.
#
# Com o percentil padrão de 99.5, o teto é o P99,5 (= 55) e o piso é o P0,5
# (= 17). O piso de 17 absorve BMIs de 12 a 16, que são fisiologicamente
# implausíveis (provável erro de digitação ou desnutrição severa) sem removê-los.

def tratar_bmi_extremo(
    df: pd.DataFrame,
    coluna: str = "BMI",
    limite_percentil: float = 99.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if coluna not in df.columns:
        raise ValueError(f"Coluna não encontrada: {coluna}")
    if not 50 < limite_percentil <= 100:
        raise ValueError("O percentil deve estar entre 50 (exclusivo) e 100.")

    teto = float(df[coluna].quantile(limite_percentil / 100))
    piso = float(df[coluna].quantile((100 - limite_percentil) / 100))

    mascara_superior = df[coluna] > teto
    mascara_inferior = df[coluna] < piso

    registros_superior = df.loc[mascara_superior].copy()
    registros_superior.insert(0, "indice_original", registros_superior.index)
    registros_superior["valor_original"] = registros_superior[coluna]
    registros_superior["valor_limitado"] = teto
    registros_superior["tipo_limite"] = "superior"

    registros_inferior = df.loc[mascara_inferior].copy()
    registros_inferior.insert(0, "indice_original", registros_inferior.index)
    registros_inferior["valor_original"] = registros_inferior[coluna]
    registros_inferior["valor_limitado"] = piso
    registros_inferior["tipo_limite"] = "inferior"

    registros = pd.concat(
        [registros_superior, registros_inferior], ignore_index=True
    )

    tratado = df.copy()
    tratado.loc[mascara_superior, coluna] = teto
    tratado.loc[mascara_inferior, coluna] = piso
    return tratado.reset_index(drop=True), registros.reset_index(drop=True)

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
    total_ajustados: int,
    pasta_saida: Path,
) -> None:
    percentual = (total_ajustados / len(antes) * 100) if len(antes) else 0.0
    comparacao = {
        "antes": {"linhas": len(antes), "colunas": antes.shape[1]},
        "depois": {"linhas": len(depois), "colunas": depois.shape[1]},
        "linhas_removidas": 0,
        "registros_com_bmi_limitado": total_ajustados,
        "percentual_de_registros_ajustados": round(percentual, 4),
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

#### EXECUÇÃO

def executar_tratamento(
    entrada: Path,
    pasta_saida: Path,
    limite_variancia: float,
    limite_percentil_bmi: float,
) -> None:
    original = ler_dataset(entrada)
    selecionado = selecionar_atributos(original)
    sem_redundancia, removidas = remover_sem_variancia(
        selecionado, limite_variancia
    )
    tratado, registros_bmi = tratar_bmi_extremo(
        sem_redundancia, limite_percentil=limite_percentil_bmi
    )

    pasta_saida.mkdir(parents=True, exist_ok=True)
    tratado.to_csv(pasta_saida / "diabetes_cardiaco_tratado.csv", index=False)
    registros_bmi.to_csv(pasta_saida / "registros_bmi_limitado.csv", index=False)
    gerar_comparacao(
        selecionado, tratado, removidas, len(registros_bmi), pasta_saida
    )

    print("Tratamento concluído.")
    print(f"Linhas antes: {len(selecionado):,}")
    print(f"Linhas depois: {len(tratado):,}")
    print(f"Registros com BMI limitado: {len(registros_bmi):,}")
    if not registros_bmi.empty:
        print(
            "Limites por tipo: "
            + ", ".join(
                f"{tipo}={int(total)}"
                for tipo, total in registros_bmi["tipo_limite"].value_counts().items()
            )
        )
    print(f"Atributos removidos por variância: {removidas or 'nenhum'}")
    print(f"Resultados: {pasta_saida.resolve()}")

#  Aceita um percentil entre 50 (exclusivo) e 100 para limitar o BMI.

def interpretar_percentil(valor: str) -> float:
   
    try:
        numero = float(valor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use um número maior que 50 e menor ou igual a 100."
        ) from exc
    if not 50 < numero <= 100:
        raise argparse.ArgumentTypeError(
            "O percentil deve ser maior que 50 e menor ou igual a 100."
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
        "--percentil-bmi", type=interpretar_percentil, default=99.5,
        help="Percentil simétrico usado como teto/piso do BMI (padrão: 99.5)",
    )
    return parser.parse_args()


def main() -> int:
    args = argumentos()
    try:
        executar_tratamento(
            args.entrada, args.saida_dir, args.limite_variancia, args.percentil_bmi
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
