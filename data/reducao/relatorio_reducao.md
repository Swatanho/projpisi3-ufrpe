# Redução de Dimensionalidade (PCA + UMAP) — HeartHealth (PISI3)

## 1. Contexto

- As 20 features do contrato já são **poucas e interpretáveis** (3 contínuas + 13 binárias + 4 ordinais). A redução de dimensionalidade foi aplicada como **análise e visualização**, não como pré-processamento obrigatório do modelo.
- **PCA** (correlação): matriz padronizada (Z-score de todas as features) para decompor a variância compartilhada.
- **UMAP** (não linear): embedding 2D para inspecionar a estrutura de separação DCV × não-DCV e o ruído do HDBSCAN.

## 2. PCA — variância explicada

- Componentes necessários: **14** p/ 80%, **17** p/ 90%, **18** p/ 95% da variância.

| componente   |   var_explicada |   var_acumulada |
|:-------------|----------------:|----------------:|
| PC1          |           0.158 |           0.158 |
| PC2          |           0.086 |           0.245 |
| PC3          |           0.068 |           0.313 |
| PC4          |           0.06  |           0.373 |
| PC5          |           0.058 |           0.431 |
| PC6          |           0.055 |           0.486 |
| PC7          |           0.049 |           0.535 |
| PC8          |           0.048 |           0.583 |
| PC9          |           0.047 |           0.63  |
| PC10         |           0.044 |           0.674 |

## 3. Interpretação dos componentes (loadings, top 3 por PC)

| componente   | feature           |   peso |
|:-------------|:------------------|-------:|
| PC1          | GenHlth           |  0.423 |
| PC1          | DiffWalk          |  0.384 |
| PC1          | PhysHlth          |  0.368 |
| PC2          | Age               |  0.483 |
| PC2          | AnyHealthcare     |  0.39  |
| PC2          | NoDocbcCost       | -0.374 |
| PC3          | Fruits            |  0.526 |
| PC3          | Veggies           |  0.496 |
| PC3          | Sex               | -0.41  |
| PC4          | Education         |  0.444 |
| PC4          | Income            |  0.407 |
| PC4          | BMI               |  0.357 |
| PC5          | Smoker            |  0.576 |
| PC5          | HvyAlcoholConsump |  0.569 |
| PC5          | Sex               |  0.299 |

## 4. Impacto no modelo (full vs. componentes principais)

| config                 |   n_dims |   roc_auc |   pr_auc |
|:-----------------------|---------:|----------:|---------:|
| 20 features (completo) |       20 |     0.834 |    0.361 |
| PCA (17 comps, 90%)    |       17 |     0.826 |    0.343 |
| PCA (18 comps, 95%)    |       18 |     0.826 |    0.347 |

## 5. Conclusão

- As 20 features **não são altamente redundantes**: para reter 90% da variância são necessários ~17 componentes — a base já é enxuta para o problema.
- A projeção 2D (PCA/UMAP) **não separa linearmente** as classes DCV × não-DCV — coerente com o AUC moderado (~0.83) e com a natureza autorrelatada do BRFSS.
- Manter as 20 features (ou reduzir para os PCs) pouco muda o modelo: a redução **não agrega poder preditivo**, apenas serve para visualização e entendimento da estrutura.

## 6. Referências
- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.