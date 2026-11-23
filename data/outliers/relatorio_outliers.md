# Relatório de Outliers Multivariados — HeartHealth (PISI3)

## 1. Como foram calculados os outliers multivariados

### Representação das variáveis
- As **3 contínuas** (`BMI`, `MentHlth`, `PhysHlth`) foram **padronizadas** (StandardScaler / Z-score) para que a magnitude (ex.: BMI ~25) não domine as ordinais/binárias na distância.
- As **13 binárias** (0/1) e **4 ordinais** (`GenHlth`, `Age`, `Education`, `Income`) entraram **como estão** (escala ordinal), preservando a ordem e sem one-hot (que inflaria a dimensionalidade artificialmente).
- Nenhuma variável foi descartada; a análise é verdadeiramente multivariada (20 features do contrato), diferente da versão anterior que usava apenas 3 contínuas.

### Isolation Forest (Liu et al., 2008)
- Isola pontos por **partições aleatórias recursivas**: quanto menos cortes são necessários para isolar um ponto, mais anômalo ele é.
- Não paramétrico, custo ~O(n log n), escala para 229 mil linhas.
- `contamination = 1%` (parâmetro explícito de taxa esperada de anomalia).
- Estratégia **não destrutiva**: gera a flag `is_outlier`, nenhum registro é descartado.

### HDBSCAN (Campello et al., 2013/2015)
- Clusterização hierárquica por **densidade**: pontos que não formam estrutura densa recebem rótulo **-1** (ruído/outlier).
- **Não exige** o hiperparâmetro de contaminação (vantagem sobre o IF).
- Complementado pelo **GLOSH** (`outlier_scores_`), escore contínuo de anomalia.
- Por custo (grafo k-NN), rodou numa **subamostra estratificada de 50 mil** linhas; o resultado é extrapolável mas documentado como tal.

## 2. Resultados

- Prevalência geral de DCV: **10.33%**.

| Método | Outliers (n) | Outliers (%) | Prev. DCV (outliers) | Prev. DCV (inliers) | Lift |
|---|---:|---:|---:|---:|---:|
| IF_3cols | 2,295 | 1.00% | 18.74% | 10.25% | 1.81x |
| IF_20cols | 2,295 | 1.00% | 29.41% | 10.14% | 2.85x |
| HDBSCAN (subamostra de 50k (rótulo -1)) | 11,912 | 23.82% | 16.88% | 8.28% | 1.63x |

## 3. Sobreposição entre métodos (Jaccard, subamostra 50k)

| par                  |   jaccard |
|:---------------------|----------:|
| IF_3cols x IF_20cols | 0.0719656 |
| IF_20cols x HDBSCAN  | 0.0441572 |
| IF_3cols x HDBSCAN   | 0.0393621 |

- O HDBSCAN identificou **2 clusters densos** e **11,912 pontos de ruído** na subamostra.

## 4. Conclusão

- Um outlier é clinicamente relevante se a prevalência de DCV nele **difere** da prevalência geral (lift ≠ 1): pontos "estranhos" no espaço das features que concentram ou diluem risco são informativos para a triagem.
- **Ampliar de 3 para 20 features melhora a detecção**: a sobreposição entre IF_3cols e IF_20cols é de apenas ~7%, e o lift sobe de 1,81x para 2,85x — o IF com 20 features encontra outliers mais associados ao risco de DCV.
- **HDBSCAN não foi o melhor detector de outlier aqui**: marcou 23,8% da subamostra como ruído (rótulo -1) e esse ruído tem lift de apenas 1,63x, inferior ao IF de 20 features (2,85x). A noção de "ruído" do HDBSCAN é densidade local, não risco clínico — em bases de saúde densas ela sinaliza uma fração grande e pouco específica de pontos.
- **Conclusão:** manter o Isolation Forest (agora com as 20 features) como método principal, com o HDBSCAN documentado como verificação complementar (pedido do orientador atendido: foi aplicado e comparado).

## 5. Referências
- Liu, F. T. et al. (2008). *Isolation Forest*. ICDM.
- Campello, R. J. G. B. et al. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
- Campello, R. J. G. B. et al. (2015). *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*. ACM TKDD.