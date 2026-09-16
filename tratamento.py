"""
Limpeza e preparação inicial dos dados.

Etapas:
1. Seleção das variáveis relevantes ao problema.
2. Verificação de valores ausentes/não numéricos.
3. Remoção de atributos sem variância.
4. Tratamento de valores extremos de BMI por winsorização.
5. Geração de metadados para explicar a base e os motivos das escolhas.

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
    "GenHlth", "DiffWalk", "Fruits", "Veggies", "AnyHealthcare",
    "NoDocbcCost", "MentHlth", "PhysHlth", "Education", "Income",
]

DICIONARIO_VARIAVEIS = {
    "HeartDiseaseorAttack": {
        "nome": "Doença cardíaca ou ataque cardíaco",
        "tipo": "alvo binário",
        "descricao": "Indica se o respondente relatou doença cardíaca coronariana ou infarto/ataque cardíaco.",
        "papel": "variável-alvo; não é usada para formar os clusters",
    },
    "HighBP": {
        "nome": "Pressão arterial alta",
        "tipo": "binária",
        "descricao": "Indicador de relato de pressão arterial alta.",
        "papel": "fator clínico associado",
    },
    "HighChol": {
        "nome": "Colesterol alto",
        "tipo": "binária",
        "descricao": "Indicador de relato de colesterol alto.",
        "papel": "fator clínico associado",
    },
    "CholCheck": {
        "nome": "Verificação de colesterol",
        "tipo": "binária",
        "descricao": "Indica realização de verificação de colesterol conforme a codificação da base.",
        "papel": "indicador de acompanhamento preventivo",
    },
    "BMI": {
        "nome": "Índice de Massa Corporal (IMC)",
        "tipo": "contínua",
        "descricao": "Medida numérica usada para representar o índice de massa corporal.",
        "papel": "característica antropométrica",
    },
    "Diabetes_012": {
        "nome": "Classificação de diabetes",
        "tipo": "categórica ordinal",
        "descricao": "0 = sem diabetes ou apenas durante a gravidez; 1 = pré-diabetes; 2 = diabetes.",
        "papel": "condição clínica associada",
    },
    "Smoker": {
        "nome": "Tabagismo",
        "tipo": "binária",
        "descricao": "Indicador relacionado ao histórico/condição de fumante conforme a codificação do conjunto.",
        "papel": "fator comportamental associado",
    },
    "HvyAlcoholConsump": {
        "nome": "Consumo elevado de álcool",
        "tipo": "binária",
        "descricao": "Indicador de consumo elevado de álcool segundo a definição da base.",
        "papel": "fator comportamental associado",
    },
    "PhysActivity": {
        "nome": "Atividade física",
        "tipo": "binária",
        "descricao": "Indicador de prática de atividade física conforme a codificação do conjunto.",
        "papel": "fator comportamental associado",
    },
    "Age": {
        "nome": "Faixa etária",
        "tipo": "ordinal",
        "descricao": "Faixas etárias codificadas em valores de 1 a 13, de 18–24 até 80+.",
        "papel": "característica demográfica",
    },
    "Sex": {
        "nome": "Sexo",
        "tipo": "binária",
        "descricao": "Categoria sexual codificada numericamente na base.",
        "papel": "característica demográfica",
    },
    "Stroke": {
        "nome": "Histórico de acidente vascular cerebral",
        "tipo": "binária",
        "descricao": "Indica histórico relatado de AVC.",
        "papel": "condição clínica associada",
    },
    "GenHlth": {
        "nome": "Saúde geral percebida",
        "tipo": "ordinal",
        "descricao": "Autoavaliação geral de saúde, de excelente a ruim.",
        "papel": "indicador de estado de saúde",
    },
    "DiffWalk": {
        "nome": "Dificuldade para caminhar",
        "tipo": "binária",
        "descricao": "Indicador de dificuldade para caminhar ou subir escadas.",
        "papel": "indicador funcional associado",
    },
    "Fruits": {
        "nome": "Consumo de frutas",
        "tipo": "binária",
        "descricao": "Indicador de consumo regular de frutas conforme a codificação da base.",
        "papel": "fator comportamental associado",
    },
    "Veggies": {
        "nome": "Consumo de vegetais",
        "tipo": "binária",
        "descricao": "Indicador de consumo regular de vegetais conforme a codificação da base.",
        "papel": "fator comportamental associado",
    },
    "AnyHealthcare": {
        "nome": "Cobertura de saúde",
        "tipo": "binária",
        "descricao": "Indica se o respondente possui qualquer tipo de cobertura de saúde.",
        "papel": "indicador de acesso a serviços de saúde",
    },
    "NoDocbcCost": {
        "nome": "Não procurou médico por custo",
        "tipo": "binária",
        "descricao": "Indica se o respondente deixou de procurar médico por questões de custo nos últimos 12 meses.",
        "papel": "indicador de barreira de acesso a saúde",
    },
    "MentHlth": {
        "nome": "Dias de saúde mental ruim",
        "tipo": "contínua (contagem)",
        "descricao": "Número de dias, nos últimos 30 dias, em que a saúde mental não foi boa.",
        "papel": "indicador de estado de saúde",
    },
    "PhysHlth": {
        "nome": "Dias de saúde física ruim",
        "tipo": "contínua (contagem)",
        "descricao": "Número de dias, nos últimos 30 dias, em que a saúde física não foi boa.",
        "papel": "indicador de estado de saúde",
    },
    "Education": {
        "nome": "Escolaridade",
        "tipo": "ordinal",
        "descricao": "Nível de escolaridade codificado de 1 (nunca frequentou escola) a 6 (faculdade ou mais).",
        "papel": "característica socioeconômica",
    },
    "Income": {
        "nome": "Faixa de renda",
        "tipo": "ordinal",
        "descricao": "Faixa de renda anual familiar codificada de 1 (< 10 mil dólares) a 8 (75 mil dólares ou mais).",
        "papel": "característica socioeconômica",
    },
}


# ---------------------------------------------------------------------------
# Carregamento


def ler_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, low_memory=False)


# ---------------------------------------------------------------------------
# Seleção e validação


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


# ---------------------------------------------------------------------------
# Tratamento de extremos


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

    registros = pd.concat([registros_superior, registros_inferior], ignore_index=True)

    tratado = df.copy()
    tratado.loc[mascara_superior, coluna] = teto
    tratado.loc[mascara_inferior, coluna] = piso
    return tratado.reset_index(drop=True), registros.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparação e documentação da base


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


def gerar_metadados_base(
    original: pd.DataFrame,
    selecionado: pd.DataFrame,
    tratado: pd.DataFrame,
    removidas: list[str],
    registros_bmi: pd.DataFrame,
    entrada: Path,
    limite_percentil_bmi: float,
    pasta_saida: Path,
) -> None:
    metadata = {
        "fonte": {
            "dataset": "Diabetes Health Indicators Dataset",
            "arquivo_utilizado": entrada.name,
            "origem": "Kaggle / BRFSS 2015",
            "url": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
            "observacao": "A base é observacional e baseada em respostas de inquérito; associações encontradas no painel não devem ser interpretadas como causalidade.",
        },
        "dimensoes": {
            "linhas_originais": int(len(original)),
            "colunas_originais": int(original.shape[1]),
            "linhas_selecionadas": int(len(selecionado)),
            "colunas_selecionadas_com_alvo": int(selecionado.shape[1]),
            "colunas_tratadas_com_alvo": int(tratado.shape[1]),
            "atributos_removidos_por_variancia": removidas,
        },
        "variavel_alvo": ALVO,
        "variaveis": {
            c: DICIONARIO_VARIAVEIS.get(c, {"nome": c, "tipo": "não documentada"})
            for c in tratado.columns
        },
        "motivos_do_tratamento": [
            {
                "etapa": "Seleção de atributos",
                "motivo": "Todas as variáveis da base original são mantidas (clínicas, demográficas, funcionais, comportamentais, socioeconômicas e de acesso à saúde), preservando a informação disponível para a análise.",
            },
            {
                "etapa": "Remoção por variância",
                "motivo": "Eliminar atributos sem informação discriminativa, evitando dimensões inúteis na análise de clusters.",
            },
            {
                "etapa": "Winsorização do BMI",
                "motivo": "Reduzir a influência de valores extremos sem descartar pessoas da base nem alterar a variável-alvo.",
                "percentil": limite_percentil_bmi,
                "registros_ajustados": int(len(registros_bmi)),
            },
            {
                "etapa": "Imputação, encoding e escala",
                "motivo": "Adequar tipos de dados, categorias e escalas para os algoritmos de Machine Learning, evitando que uma variável numérica domine a distância usada pelo K-Means.",
            },
        ],
    }
    (pasta_saida / "metadados_base.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Execução


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
    gerar_metadados_base(
        original,
        selecionado,
        tratado,
        removidas,
        registros_bmi,
        entrada,
        limite_percentil_bmi,
        pasta_saida,
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
