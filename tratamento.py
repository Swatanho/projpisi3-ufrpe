"""
Limpeza e preparação inicial dos dados de saúde.

O tratamento mantém as variáveis necessárias para o painel exploratório e
para etapas futuras de Machine Learning.

Etapas:
1. Seleção das variáveis relevantes.
2. Validação de tipos e valores ausentes.
3. Remoção de atributos sem variância.
4. Winsorização dos extremos de BMI.
5. Criação de Diabetes_binary a partir de Diabetes_012, quando necessário.
6. Geração de metadados e estatísticas antes/depois.

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
    "GenHlth", "DiffWalk", "AnyHealthcare", "NoDocbcCost", "Income", "Education",
]

DICIONARIO_VARIAVEIS = {
    "HeartDiseaseorAttack": {
        "nome": "Doença cardíaca ou ataque cardíaco",
        "tipo": "binária",
        "descricao": "Indica relato de doença cardíaca coronariana ou ataque cardíaco.",
        "papel": "desfecho observado",
    },
    "Sex": {
        "nome": "Sexo",
        "tipo": "binária",
        "descricao": "0 = mulher; 1 = homem, conforme a codificação usada no painel.",
        "papel": "variável de comparação demográfica",
    },
    "Smoker": {
        "nome": "Tabagismo",
        "tipo": "binária",
        "descricao": "Indicador de histórico/condição de fumante conforme a definição da base.",
        "papel": "fator comportamental",
    },
    "HvyAlcoholConsump": {
        "nome": "Consumo elevado de álcool",
        "tipo": "binária",
        "descricao": "Indicador de consumo elevado de álcool segundo a codificação do conjunto.",
        "papel": "fator comportamental",
    },
    "PhysActivity": {
        "nome": "Atividade física",
        "tipo": "binária",
        "descricao": "Indicador de prática de atividade física; o painel também deriva a categoria sem atividade física para facilitar a leitura.",
        "papel": "fator comportamental/protetor",
    },
    "CholCheck": {
        "nome": "Verificação de colesterol",
        "tipo": "binária",
        "descricao": "Indicador de ter realizado verificação de colesterol nos últimos 5 anos.",
        "papel": "comportamento preventivo",
    },
    "AnyHealthcare": {
        "nome": "Cobertura de saúde",
        "tipo": "binária",
        "descricao": "Indica se a pessoa possui algum tipo de cobertura/plano de saúde.",
        "papel": "acesso/prevenção",
    },
    "NoDocbcCost": {
        "nome": "Barreira financeira ao atendimento",
        "tipo": "binária",
        "descricao": "Indica se a pessoa deixou de procurar um médico por causa do custo.",
        "papel": "barreira de acesso",
    },
    "Income": {
        "nome": "Faixa de renda",
        "tipo": "ordinal",
        "descricao": "Categoria ordinal de renda codificada de 1 a 8; valores maiores correspondem a faixas de renda maiores.",
        "papel": "determinante socioeconômico",
    },
    "Education": {
        "nome": "Escolaridade",
        "tipo": "ordinal",
        "descricao": "Categoria ordinal de escolaridade codificada de 1 a 6; valores maiores correspondem a maior escolaridade.",
        "papel": "determinante socioeconômico",
    },
    "GenHlth": {
        "nome": "Saúde geral percebida",
        "tipo": "ordinal",
        "descricao": "Autoavaliação de saúde de 1 = excelente a 5 = ruim.",
        "papel": "desfecho de saúde percebida",
    },
    "Diabetes_012": {
        "nome": "Classificação de diabetes",
        "tipo": "ordinal",
        "descricao": "0 = sem diabetes, 1 = pré-diabetes, 2 = diabetes.",
        "papel": "condição clínica",
    },
    "Diabetes_binary": {
        "nome": "Diabetes binário",
        "tipo": "binária",
        "descricao": "Derivada de Diabetes_012: 0 = sem diabetes; 1 = pré-diabetes ou diabetes.",
        "papel": "desfecho clínico para análise socioeconômica",
    },
    "HighBP": {
        "nome": "Pressão arterial alta",
        "tipo": "binária",
        "descricao": "Indicador de pressão arterial alta.",
        "papel": "fator clínico",
    },
    "HighChol": {
        "nome": "Colesterol alto",
        "tipo": "binária",
        "descricao": "Indicador de colesterol alto.",
        "papel": "fator clínico",
    },
    "BMI": {
        "nome": "Índice de Massa Corporal (IMC)",
        "tipo": "contínua",
        "descricao": "Medida numérica do índice de massa corporal.",
        "papel": "característica antropométrica",
    },
    "Age": {
        "nome": "Faixa etária",
        "tipo": "ordinal",
        "descricao": "Faixas etárias codificadas de 1 a 13.",
        "papel": "característica demográfica",
    },
    "Stroke": {
        "nome": "Histórico de AVC",
        "tipo": "binária",
        "descricao": "Indicador de histórico relatado de AVC.",
        "papel": "condição clínica",
    },
    "DiffWalk": {
        "nome": "Dificuldade para caminhar",
        "tipo": "binária",
        "descricao": "Indicador de dificuldade para caminhar ou subir escadas.",
        "papel": "indicador funcional",
    },
}


def ler_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, low_memory=False)


def selecionar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    ausentes = [c for c in [ALVO, *ATRIBUTOS_RELEVANTES] if c not in df.columns]
    if ausentes:
        dica = " Use o arquivo diabetes_012_health_indicators_BRFSS2015.csv do dataset informado."
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(ausentes) + "." + dica)

    colunas = [ALVO, *ATRIBUTOS_RELEVANTES]
    selecionado = df.loc[:, colunas].copy()
    for coluna in colunas:
        selecionado[coluna] = pd.to_numeric(selecionado[coluna], errors="coerce")

    # O arquivo principal do Kaggle traz Diabetes_012. Criamos Diabetes_binary
    # para o terceiro gráfico, sem depender de uma segunda versão do dataset.
    selecionado["Diabetes_binary"] = (selecionado["Diabetes_012"] > 0).astype(int)

    faltantes = selecionado.isna().sum()
    faltantes = faltantes[faltantes > 0]
    if not faltantes.empty:
        detalhes = ", ".join(f"{c}={int(n)}" for c, n in faltantes.items())
        raise ValueError("Há valores ausentes ou não numéricos. Corrija-os antes: " + detalhes)
    return selecionado


def remover_sem_variancia(df: pd.DataFrame, limite_variancia: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    if limite_variancia < 0:
        raise ValueError("O limite de variância não pode ser negativo.")
    preditores = df.drop(columns=ALVO)
    seletor = VarianceThreshold(threshold=limite_variancia)
    seletor.fit(preditores)
    mantidas = preditores.columns[seletor.get_support()].tolist()
    removidas = [c for c in preditores.columns if c not in mantidas]
    return df.loc[:, [ALVO, *mantidas]].copy(), removidas


def tratar_bmi_extremo(df: pd.DataFrame, coluna: str = "BMI", limite_percentil: float = 99.5) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def resumo_colunas(df: pd.DataFrame, etapa: str) -> pd.DataFrame:
    resumo = df.describe(include="all").T.reset_index(names="coluna")
    resumo.insert(0, "etapa", etapa)
    resumo["valores_ausentes"] = [int(df[c].isna().sum()) for c in resumo["coluna"]]
    resumo["valores_unicos"] = [int(df[c].nunique(dropna=False)) for c in resumo["coluna"]]
    return resumo


def gerar_comparacao(antes: pd.DataFrame, depois: pd.DataFrame, removidas: list[str], total_ajustados: int, pasta_saida: Path) -> None:
    percentual = (total_ajustados / len(antes) * 100) if len(antes) else 0.0
    comparacao = {
        "antes": {"linhas": len(antes), "colunas": antes.shape[1]},
        "depois": {"linhas": len(depois), "colunas": depois.shape[1]},
        "linhas_removidas": 0,
        "registros_com_bmi_limitado": total_ajustados,
        "percentual_de_registros_ajustados": round(percentual, 4),
        "atributos_removidos_por_variancia": removidas,
        "distribuicao_alvo_antes": {str(k): int(v) for k, v in antes[ALVO].value_counts().sort_index().items()},
        "distribuicao_alvo_depois": {str(k): int(v) for k, v in depois[ALVO].value_counts().sort_index().items()},
    }
    (pasta_saida / "comparacao_antes_depois.json").write_text(json.dumps(comparacao, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat([resumo_colunas(antes, "antes"), resumo_colunas(depois, "depois")], ignore_index=True).to_csv(
        pasta_saida / "estatisticas_antes_depois.csv", index=False
    )


def gerar_metadados_base(original: pd.DataFrame, selecionado: pd.DataFrame, tratado: pd.DataFrame, removidas: list[str], registros_bmi: pd.DataFrame, entrada: Path, limite_percentil_bmi: float, pasta_saida: Path) -> None:
    metadata = {
        "fonte": {
            "dataset": "Diabetes Health Indicators Dataset",
            "arquivo_utilizado": entrada.name,
            "origem": "Kaggle / BRFSS 2015",
            "url": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
            "observacao": "Base observacional. As comparações do painel descrevem associações na amostra e não demonstram causalidade.",
        },
        "dimensoes": {
            "linhas_originais": int(len(original)),
            "colunas_originais": int(original.shape[1]),
            "linhas_selecionadas": int(len(selecionado)),
            "colunas_selecionadas": int(selecionado.shape[1]),
            "colunas_tratadas": int(tratado.shape[1]),
            "atributos_removidos_por_variancia": removidas,
        },
        "variaveis": {
            c: DICIONARIO_VARIAVEIS.get(c, {"nome": c, "tipo": "não documentada"})
            for c in tratado.columns
        },
        "motivos_do_tratamento": [
            {"etapa": "Seleção de atributos", "motivo": "Manter somente as variáveis necessárias para as análises de gênero, prevenção, acesso e condições socioeconômicas."},
            {"etapa": "Diabetes_binary", "motivo": "Criar uma variável binária simples para comparar presença/ausência de diabetes no eixo socioeconômico."},
            {"etapa": "Remoção por variância", "motivo": "Retirar atributos sem variação, caso existam, evitando informação inútil na análise."},
            {"etapa": "Winsorização do BMI", "motivo": "Reduzir a influência de valores extremos sem excluir respondentes."},
            {"etapa": "Validação de dados", "motivo": "Garantir valores numéricos e ausência de campos faltantes antes das análises."},
        ],
    }
    (pasta_saida / "metadados_base.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def executar_tratamento(entrada: Path, pasta_saida: Path, limite_variancia: float, limite_percentil_bmi: float) -> None:
    original = ler_dataset(entrada)
    selecionado = selecionar_atributos(original)
    sem_redundancia, removidas = remover_sem_variancia(selecionado, limite_variancia)
    tratado, registros_bmi = tratar_bmi_extremo(sem_redundancia, limite_percentil=limite_percentil_bmi)

    pasta_saida.mkdir(parents=True, exist_ok=True)
    tratado.to_csv(pasta_saida / "diabetes_cardiaco_tratado.csv", index=False)
    registros_bmi.to_csv(pasta_saida / "registros_bmi_limitado.csv", index=False)
    gerar_comparacao(selecionado, tratado, removidas, len(registros_bmi), pasta_saida)
    gerar_metadados_base(original, selecionado, tratado, removidas, registros_bmi, entrada, limite_percentil_bmi, pasta_saida)

    print("Tratamento concluído.")
    print(f"Linhas: {len(tratado):,}")
    print(f"Colunas tratadas: {tratado.shape[1]}")
    print(f"BMI ajustado: {len(registros_bmi):,}")
    print(f"Atributos removidos por variância: {removidas or 'nenhum'}")
    print(f"Resultados: {pasta_saida.resolve()}")


def interpretar_percentil(valor: str) -> float:
    try:
        numero = float(valor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use um número maior que 50 e menor ou igual a 100.") from exc
    if not 50 < numero <= 100:
        raise argparse.ArgumentTypeError("O percentil deve ser maior que 50 e menor ou igual a 100.")
    return numero


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path, nargs="?", default=Path("DB/diabetes_012_health_indicators_BRFSS2015.csv"))
    parser.add_argument("--saida-dir", type=Path, default=Path("DB/dados_tratados"))
    parser.add_argument("--limite-variancia", type=float, default=0.0)
    parser.add_argument("--percentil-bmi", type=interpretar_percentil, default=99.5)
    return parser.parse_args()


def main() -> int:
    args = argumentos()
    try:
        executar_tratamento(args.entrada, args.saida_dir, args.limite_variancia, args.percentil_bmi)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
