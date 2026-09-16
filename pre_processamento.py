"""
Pré-processamento dos dados para segmentação e análise preditiva.
Realiza:
1. Imputação de Valores Ausentes (Mediana e Moda)
2. One-Hot Encoding
3. Padronização/Normalização de Escala
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Variável Alvo (não deve ser escalada)
ALVO = "HeartDiseaseorAttack"

# Separação das Features com base no comportamento
# O script analise_clusters.py espera que o Diabetes seja convertido em 3 colunas via OHE
VARIAVEIS_CONTINUAS = ["BMI", "MentHlth", "PhysHlth"]
VARIAVEIS_NOMINAIS = ["Diabetes_012"]
VARIAVEIS_BINA_ORDINAIS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "HvyAlcoholConsump",
    "PhysActivity", "Age", "Sex", "Stroke", "GenHlth", "DiffWalk",
    "Fruits", "Veggies", "AnyHealthcare", "NoDocbcCost",
    "Education", "Income"
]

def main() -> int:
    entrada = Path("DB/dados_tratados/diabetes_cardiaco_tratado.csv")
    saida = Path("DB/dados_tratados/diabetes_cardiaco_pre_processado.csv")
    
    if not entrada.exists():
        print(f"ERRO: Arquivo {entrada} não encontrado. Execute o tratamento.py primeiro.")
        return 1

    print("Carregando os dados tratados...")
    df = pd.read_csv(entrada)
    y = df[ALVO]
    X = df.drop(columns=[ALVO])

    # 1. Pipeline para Variáveis Contínuas (Imputação por Mediana + StandardScaler)
    pipe_continua = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Pipeline para Variáveis Nominais (Imputação por Moda + One-Hot Encoding)
    # A regressão logística ou K-means necessitam que variáveis nominais sejam explícitas
    pipe_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    # 3. Pipeline para Variáveis Ordinais e Binárias (Imputação por Moda + MinMaxScaler)
    # Mantém a escala entre 0 e 1, preservando a lógica de limites bem definidos
    pipe_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", MinMaxScaler())
    ])

    # 4. Agrupando os pipelines no ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", pipe_continua, VARIAVEIS_CONTINUAS),
            ("nom", pipe_nominal, VARIAVEIS_NOMINAIS),
            ("ord", pipe_ordinal, VARIAVEIS_BINA_ORDINAIS)
        ],
        remainder="drop" # Remove qualquer coluna extra acidental
    )

    print("Aplicando imputação, encoding e escalonamento...")
    X_processado = preprocessor.fit_transform(X)
    
    # 5. Recuperar os nomes das colunas após o processamento
    # O ColumnTransformer adiciona prefixos (ex: num__BMI). Precisamos limpá-los.
    colunas_out = preprocessor.get_feature_names_out()
    colunas_limpas = [c.split("__")[-1] for c in colunas_out]

    # 6. Remontando o DataFrame final
    df_processado = pd.DataFrame(X_processado, columns=colunas_limpas, index=df.index)
    df_processado.insert(0, ALVO, y) # Devolvemos a variável alvo intacta

    # 7. Salvando o resultado
    saida.parent.mkdir(parents=True, exist_ok=True)
    df_processado.to_csv(saida, index=False)
    
    print("Pré-processamento concluído com sucesso!")
    print(f"Total de colunas geradas: {df_processado.shape[1]}")
    print(f"Salvo em: {saida.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
