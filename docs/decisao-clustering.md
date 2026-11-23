# Decisão de Técnica de Clustering — HeartHealth (PISI3)

**Data:** 20/10/2026 · **Área:** ML · **Status:** Decidido e implementado (branch `dev-rebecca`, `src/clustering_teste.py`); ajustes de alinhamento mapeados (§7)

## 1. Pergunta da planilha

*Qual técnica de clustering será usada no projeto e por quê?*

## 2. Decisão

| Finalidade | Técnica escolhida | Resumo do motivo |
|---|---|---|
| **Segmentação de perfis** | **K-Prototypes** (`kmodes`) | Base com tipos mistos: 3 numéricas contínuas + 13 binárias + 4 ordinais |
| **Detecção de outliers** | **Isolation Forest** (mantido) | Não destrutivo (flag `is_outlier`), escala bem para 229 mil linhas |

O **DBSCAN** (sugestão do professor) foi avaliado e **rejeitado** para ambas as finalidades — justificativa nos §4 e §5.

## 3. Por que K-Prototypes

- O dataset (`diabetes_binary_health_indicators_BRFSS2015`) é de **tipos mistos**. O K-Means padrão exige distância euclidiana: aplicar one-hot nas 13 variáveis binárias faria a distância ser dominada por dimensões artificiais, perdendo interpretabilidade.
- O K-Prototypes (Huang, 1998) combina **distância euclidiana** para as numéricas escaladas com **dissimilaridade de matching (tipo Hamming)** para as categóricas, ponderadas por γ — exatamente o formato do nosso problema.
- **Features usadas:** as mesmas **20 variáveis do contrato do app (DSI)**, exceto o alvo `HeartDiseaseorAttack` (evita leakage). `Diabetes_binary` **excluído**, pois não faz parte do contrato de features que o app envia.
- **Tratamento por tipo:**
  - Contínuas (`BMI`, `MentHlth`, `PhysHlth`): `StandardScaler`;
  - Binárias (13): categóricas (matching);
  - Ordinais (`GenHlth`, `Age`, `Education`, `Income`): implementação atual trata como **categóricas**; a alternativa (numéricas + MinMax, preservando a ordem) fica documentada como decisão aberta — ambas são defensáveis, o importante é manter consistência com o modelo supervisionado.
- **Escolha do k:** método do cotovelo (curva de custo) + silhueta em amostra; k final deve ser **derivado** dessas curvas, não fixado.

## 4. Alternativas avaliadas e rejeitadas

| Técnica | Motivo da rejeição |
|---|---|
| K-Means (+ one-hot) | Distância euclidiana inadequada para binárias; esparsidade destrói o significado dos centróides |
| GMM | Pressupõe gaussianidade por componente; variáveis binárias/ordinais violam a premissa |
| **DBSCAN** | Ver §4.1 |
| HDBSCAN | Mesmo problema de densidade, custo computacional maior |
| K-Medoids + Gower | Adequado em teoria para dados mistos, mas custo ~O(k·n²) inviável para 229 mil linhas |

### 4.1 DBSCAN (sugestão do professor) — análise

- Dados de saúde em bases grandes são **densos e com separação fraca entre grupos**: o DBSCAN tende a formar 1 cluster gigante + tudo o mais como "ruído", sem segmentação útil.
- Calibrar `eps`/`minPts` em ~20 dimensões mistas é instável (a noção de "vizinhança" depende de encoding e escala).
- Custo e ausência de `k` explícito dificultam a entrega ("quais perfis existem?").

## 5. DBSCAN × Isolation Forest (para outliers)

- DBSCAN detecta anomalias apenas como **subproduto** (pontos de ruído); Isolation Forest é **específico** para anomalia (isola pontos raros por divisões aleatórias), lida com dimensionalidade mista e roda em O(n log n).
- Estratégia adotada no `src/data_prep.py` é **não destrutiva**: nenhum registro é descartado; o IF gera a flag `is_outlier` — **2.294 registros (1,0%)** da base tratada — permitindo comparar o modelo supervisionado com/sem outliers.
- **Conclusão:** manter Isolation Forest. Uma comparação opcional DBSCAN × IF em amostra de 5–10 mil linhas pode ser executada como anexo, se o professor pedir.

## 6. Resultado preliminar (amostra de 5.000 linhas, branch `dev-rebecca`)

A segmentação já produz **gradiente de risco** claro, validando a técnica:

| Cluster | n | IMC médio | MentHlth (dias) | PhysHlth (dias) | Prevalência DCV |
|---|---|---|---|---|---|
| 0 | 2.814 | 25,1 | 1,1 | 0,9 | **5,7%** |
| 2 | 1.397 | 34,2 | 1,1 | 1,5 | **11,5%** |
| 3 | 366 | 30,5 | 26,3 | 13,5 | **19,1%** |
| 1 | 423 | 29,8 | 2,9 | 26,0 | **22,5%** |

Para referência: prevalência geral de DCV na base tratada = **10,33%** (desbalanceamento ~9:1, relevante para o modelo supervisionado).

## 7. Ajustes mapeados para após o merge da `dev-rebecca`

1. Rodar o clustering sobre o **parquet tratado** (a implementação atual lê o CSV bruto, ignorando deduplicação, winsorização e flags).
2. **Remover `Diabetes_binary`** das features (alinhamento com o contrato de 20 features do app).
3. **Derivar k** do cotovelo/silhueta (hoje k=4 fixo).
4. **Gerar a interpretação dos perfis a partir dos resultados reais** (o texto atual é fixo e diverge parcialmente dos números — ex.: "Cluster 2 com saúde física comprometida" quando PhysHlth = 1,5 dia).
5. Executar a rodada final na **base completa** (229.474 linhas).
6. Integrar os gráficos ao dashboard em aba própria (Plotly) com upgrades: prevalência com linha de base e *lift*, composição clínica por cluster, radar de perfis.

## 8. Referências

- Huang, Z. (1998). *Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values*. Data Mining and Knowledge Discovery.
- Liu, F. T. et al. (2008). *Isolation Forest*. ICDM.
- Biblioteca `kmodes`: https://github.com/nicodv/kmodes
