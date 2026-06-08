# Refatoração: Análise de Ablação + KNN Melhorado

## Contexto

Durante a modelagem supervisionada, foi identificado que a **Árvore de Decisão teve um desempenho suspeitosamente alto** (R² ≈ 0,998 em regressão, F1 ≈ 0,998 em classificação binarizada). Isso sugere possível **data leakage** — a árvore pode estar aprendendo uma feature que sintetiza o label, em vez de padrões reais.

Em paralelo, o **KNN teve desempenho fraco** (F1 baixo), porque é mais honesto com dados desbalanceados mas sofre com o espaço de features.

## Solução: Dois Experimentos

### 1. Análise de Ablação (Feature Importance)

**Objetivo**: Identificar qual feature é responsável pelo leakage.

**Método**:
- Treina os três modelos (KNN, Árvore, Random Forest) com **todas as features**
- Treina os mesmos modelos **removendo cada feature uma de cada vez**
- Compara F1 e outras métricas

**Hipótese**: Se remover uma feature específica causa queda grande em F1 (especialmente na Árvore), está confirmado o leakage.

**Arquivos**:
- `src/ablacao.py` — implementação do experimento
- `notebooks/05c_analise_ablacao_knn.ipynb` — orquestração e visualizações
- `data/interim/ablacao_resultados.parquet` — resultados

**Como executar**:
```python
from src.ablacao import main
results = main()  # Roda todas as 7 combinações (baseline + 6 ablações)
```

**Saída esperada**:
```
Tabela comparativa com 7 linhas:
1. Baseline (todas as features)
2. Sem log_valor_licitacao
3. Sem n_participantes
4. Sem n_itens
5. Sem hhi
6. Sem top1_share
7. Sem valor_total_itens

Cada linha tem métricas para KNN, Árvore e Random Forest.
```

### 2. KNN Melhorado com Tuning Agressivo

**Problema**: KNN teve F1 baixo porque:
- k foi testado apenas em {3, 5, 7, 10, 15, 20} (faltaram valores extremos)
- `weights='uniform'` não dá peso maior a vizinhos próximos
- Não houve busca de threshold ótimo em `predict_proba`

**Solução**:
- Testar k ∈ {1, 3, 5, 7, 15, 25} (mais aggressivo)
- Testar `weights` ∈ {'uniform', 'distance'}
- Encontrar threshold ótimo que maximize F1 ou PR-AUC
- Usar esse threshold em `predict_proba` em vez do padrão 0.5

**Arquivos**:
- `src/knn_melhorado.py` — implementação
- `notebooks/05c_analise_ablacao_knn.ipynb` — orquestração e visualizações
- `data/interim/knn_tuning_history.parquet` — histórico de CV
- `data/interim/knn_melhorado_resultados.png` — gráficos (ROC, PR-curve, etc)
- `data/interim/knn_comparacao_before_after.parquet` — comparação quantificada

**Como executar**:
```python
from src.knn_melhorado import run_knn_improved
result = run_knn_improved(X_train_scaled, y_train, X_test_scaled, y_test)
```

**Saída esperada**:
```
Top 5 configurações por F1:
k=7, weights='distance', F1=0.78
k=5, weights='distance', F1=0.76
k=3, weights='distance', F1=0.72
...

Threshold ótimo: 0.35 (em vez de 0.5)
F1 com threshold ótimo: 0.78 (+0.12 vs padrão)
```

## O Notebook 05c

**`notebooks/05c_analise_ablacao_knn.ipynb`** orquestra os dois experimentos:

1. **Seção 1-2**: Prepara dados
2. **Seção 2-3**: Executa ablação completa (leva ~5-10 min)
3. **Seção 4**: Visualiza impacto de cada feature
4. **Seção 5-6**: Executa KNN melhorado (leva ~3-5 min)
5. **Seção 7**: Compara KNN default vs melhorado (before/after)

## Interpretação dos Resultados

### Se a Árvore cai muito ao remover uma feature:

```
Baseline:  DT F1 = 0.998
Sem X:     DT F1 = 0.50      ← Queda MASSIVE
Sem Y:     DT F1 = 0.98      ← Queda pequena
```

→ Conclusão: Feature X é o leakage. Remover ou redesenhar.

### Se KNN melhora com tuning:

```
KNN Default (k=5):     F1 = 0.45
KNN Tuned (k=7, dist): F1 = 0.62      ← +35% de melhoria
```

→ Conclusão: KNN estava com hiperparâmetros ruins. Tuning agressivo vale a pena.

## Próximos Passos

1. **Rodar `05c_analise_ablacao_knn.ipynb`** no Jupyter
2. **Analisar tabelas de ablação**:
   - Se encontrar leakage massivo, remover a feature e retreinar tudo
   - Se não encontrar, features estão limpas
3. **Usar KNN melhorado** na seção 05_modelagem_supervisionada (substituir com `weights='distance'` e threshold ótimo)
4. **Documentar no relatório**: "Ablação não encontrou leakage" ou "Removemos feature X que causava leakage"

## Arquivos Novos

```
src/
  ├── ablacao.py                          (nova)
  ├── knn_melhorado.py                   (nova)

notebooks/
  ├── 05c_analise_ablacao_knn.ipynb     (nova)

data/interim/
  ├── ablacao_resultados.parquet        (gerado ao rodar)
  ├── ablacao_impacto_features.png      (gerado ao rodar)
  ├── knn_tuning_history.parquet        (gerado ao rodar)
  ├── knn_melhorado_resultados.png      (gerado ao rodar)
  ├── knn_comparacao_before_after.parquet (gerado ao rodar)
```

## Como Rodar Tudo

```bash
# Terminal PowerShell na raiz do projeto
.venv\Scripts\Activate.ps1

# Abrir Jupyter com o kernel certo
jupyter notebook

# Abrir: notebooks/05c_analise_ablacao_knn.ipynb
# Kernel: predicao-venv
# Rodar todas as células em ordem

# Resultado: 5 gráficos + 3 parquets gerados em data/interim/
```

## Métricas Comparadas

- **F1**: Harmônica de precision e recall (importante com desbalanceamento)
- **ROC-AUC**: Área sob a curva ROC (invariante ao threshold)
- **PR-AUC**: Área sob a curva Precision-Recall (foco em classe rara)
- **Precision**: % de predições positivas que eram verdadeiras
- **Recall**: % de anomalias detectadas

## Observações Importantes

1. **Ablação é computacionalmente cara**: Cada experimento treina 3 modelos com CV, então 7 experimentos × 3 modelos × 5 folds = 105 modelos no total. Esperado: 10-15 minutos.

2. **KNN é sensível a escala**: Por isso StandardScaler é aplicado. Nunca treinar KNN sem padronização.

3. **Threshold ótimo não é transferível**: O threshold encontrado no teste é específico para esse conjunto. Em produção, reusar com cautela.

4. **Random Forest costuma ser robusto**: Não é muito afetado por leakage porque tem regularização (max_depth, min_samples_split). Árvore simples não tem.
