import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Definir caminhos relativos com base na localização de src/
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "tratados" / "diabetes_processed.parquet"
)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza a limpeza inicial com a remoção de duplicatas exatas."""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    return df_clean


def handle_bmi_winsorization(
    df: pd.DataFrame,
    lower_percentile: float = 0.001,
    upper_percentile: float = 0.995,
) -> pd.DataFrame:
    """Aplica Winsorização na variável BMI usando limites de percentil."""
    df_out = df.copy()
    lower_limit = df_out["BMI"].quantile(lower_percentile)
    upper_limit = df_out["BMI"].quantile(upper_percentile)

    df_out["BMI"] = np.clip(df_out["BMI"], lower_limit, upper_limit)
    return df_out


def detect_multivariate_outliers(
    df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42
) -> pd.DataFrame:
    """Sinaliza anomalias multivariadas utilizando o algoritmo Isolation Forest."""
    df_out = df.copy()
    features_to_check = ["BMI", "PhysHlth", "MentHlth"]

    iso = IsolationForest(
        contamination=contamination, random_state=random_state, n_jobs=-1
    )
    preds = iso.fit_predict(df_out[features_to_check])

    df_out["is_outlier"] = preds == -1
    return df_out


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria apenas os dois atributos sintéticos oficiais definidos na Seção 4.1 do artigo."""
    df_fe = df.copy()

    # Identifica a coluna de diabetes
    diabetes_col = (
        "Diabetes_binary"
        if "Diabetes_binary" in df_fe.columns
        else "Diabetes_012"
    )

    # 1. Índice de Comorbidades Metabólicas (Somatório de Fatores Clínicos)
    df_fe["Comorbidity_Index"] = (
        df_fe["HighBP"]
        + df_fe["HighChol"]
        + (df_fe[diabetes_col] > 0).astype(int)
        + df_fe["Smoker"]
    )

    # 2. Carga de Saúde Comprometida (Somatório de Dias com Saúde Física/Mental Afetada)
    df_fe["Health_Burden_Index"] = df_fe["PhysHlth"] + df_fe["MentHlth"]

    return df_fe


def run_pipeline(
    input_path: Path = RAW_DATA_PATH, output_path: Path = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """Executa a pipeline completa de tratamento e salva o arquivo em data/tratados/."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado em: {input_path}"
        )

    print(f" Lendo arquivo original em: {input_path}")
    df_raw = pd.read_csv(input_path)
    qtd_inicial = len(df_raw)

    # Execução sequencial das etapas
    df_clean = basic_cleaning(df_raw)
    qtd_pos_duplicatas = len(df_clean)

    df_clean = handle_bmi_winsorization(df_clean)
    df_clean = detect_multivariate_outliers(df_clean)
    df_processed = apply_feature_engineering(df_clean)

    # Garantir que a pasta de destino (data/tratados/) existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvar em formato Parquet
    df_processed.to_parquet(output_path, index=False)

    # Exibição de Métricas
    duplicatas_removidas = qtd_inicial - qtd_pos_duplicatas
    pct_removido = (duplicatas_removidas / qtd_inicial) * 100
    outliers_detectados = df_processed["is_outlier"].sum()
    pct_outliers = (outliers_detectados / len(df_processed)) * 100

    print("\n" + "=" * 55)
    print(" RESUMO DA PIPELINE DE TRATAMENTO E FEATURE ENGINEERING")
    print("=" * 55)
    print(f"• Registros originais no CSV   : {qtd_inicial:,}")
    print(
        f"• Duplicatas removidas          : {duplicatas_removidas:,} ({pct_removido:.2f}%)"
    )
    print(f"• Registros válidos mantidos    : {qtd_pos_duplicatas:,}")
    print(
        f"• Outliers sinalizados (Flag)   : {outliers_detectados:,} ({pct_outliers:.2f}%)"
    )
    print(
        f"• Atributos sintéticos criados  : Comorbidity_Index, Health_Burden_Index, Lifestyle_Risk_Score"
    )
    print(f"• Arquivo final salvo em        : {output_path}")
    print("=" * 55)

    return df_processed


if __name__ == "__main__":
    # Execução direta do módulo
    run_pipeline()