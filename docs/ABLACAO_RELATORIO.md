# Experimento de Ablação: Feature `log_razao_valor_mediana`

## Resumo Executivo

Foi realizado um experimento de ablação para avaliar o impacto da feature `log_razao_valor_mediana` na detecção de anomalias em licitações. Os resultados demonstram que essa feature é **crítica** para a performance do modelo.

### Achados Principais

| Métrica | Sem Feature (Baseline) | Com Feature | Melhoria |
|---------|:--------------------:|:----------:|:---------:|
| **ROC-AUC** | 0.8398 | **0.9937** | ↑ 18.3% |
| **PR-AUC** | 0.3954 | **0.9094** | ↑ **130.0%** |
| **F1-Score** | 0.0547 | **0.5006** | ↑ **814.9%** |
| **Precision** | 0.0284 | **0.3455** | ↑ **+1,118%** |
| **Recall** | 0.7717 | **0.9087** | ↑ 17.8% |
| **Balanced Accuracy** | 0.7378 | **0.9447** | ↑ 28.0% |

---

## Detalhes do Experimento

### Configuração

**Dataset:** 197,696 licitações (2022-2023)

**Split:** 
- Treino: 158,156 (1,752 positivos = 1.11%)
- Teste: 39,540 (438 positivos = 1.11%)

**Classe desequilibrada:** 1.11% de anomalias (critério: fracionamento ou alto valor sem competição)

### Modelo

- **Algoritmo:** Random Forest (100 estimadores, profundidade 12)
- **Preprocessamento:** Imputação + Normalização (numéricas), OneHotEncoding (categóricas)
- **Balanceamento:** class_weight="balanced"
- **Validação:** Split estratificado (mantém proporção de positivos)

### Features Utilizadas

#### Baseline (9 features)
- **Numéricas:** n_participantes, n_itens, valor_total_itens, hhi, top1_share, dia_semana
- **Categóricas:** uf, situacao_licitacao, ano

#### Com log_razao_valor_mediana (10 features)
Baseline + `log_razao_valor_mediana`

**Definição da feature:**
```python
log_razao_valor_mediana = log₁₀(valor_licitacao / mediana_por_modalidade)
```
Captura desvios do valor esperado para cada modalidade de compra.

---

## Análise de Resultados

### Métricas de Classificação

#### 1. **ROC-AUC: 0.8398 → 0.9937 (+18.3%)**
- Modelo baseline já discrimina bem entre normais e anomalias (0.84)
- Adição de feature leva a discriminação quase perfeita (0.99)
- Melhoria substancial, mas não revolucionária nessa métrica

#### 2. **PR-AUC: 0.3954 → 0.9094 (+130.0%) 🎯 MAIOR IMPACTO**
- **Métrica mais relevante para dados desbalanceados**
- Baseline tem precisão muito baixa em predições positivas
- Com feature, o modelo identifica anomalias com confiança alta
- **Aumento de 2.3x na área sob a curva de precisão-recall**

#### 3. **F1-Score: 0.0547 → 0.5006 (+814.9%)**
- Baseline: quase não consegue balancear precision e recall
- Com feature: atinge equilíbrio bom (50% de F1)
- **Melhoria radical** — feature permite trade-off viável

#### 4. **Precision: 0.0284 → 0.3455 (+1,118%)**
- **Melhoria mais espetacular em termos percentuais**
- Baseline: apenas 2.8% de confiabilidade em positivos preditos
- Com feature: 34.5% de confiabilidade
- **Reduz drasticamente falsos positivos**

#### 5. **Recall: 0.7717 → 0.9087 (+17.8%)**
- Baseline já captura 77% das anomalias
- Com feature: captura 91%
- Aumento moderado, mas importante para detecção

#### 6. **Balanced Accuracy: 0.7378 → 0.9447 (+28.0%)**
- Avalia performance em ambas as classes igualmente
- Melhoria consistente de 28 pontos percentuais

### Matrizes de Confusão

**Baseline (sem log_razao_valor_mediana):**
```
               Pred Normal    Pred Suspeita
  Real Normal       27,524         11,578  (75% FPR)
  Real Suspeita        100            338
```
- **Muito ruim:** 11,578 falsos positivos entre 39,102 predições positivas

**Com log_razao_valor_mediana:**
```
               Pred Normal    Pred Suspeita
  Real Normal       38,348            754   (2% FPR)
  Real Suspeita         40            398
```
- **Muito melhor:** apenas 754 falsos positivos entre 1,152 predições positivas
- Redução de **93.5% em taxa de falso positivo**

---

## Interpretação

### Por que `log_razao_valor_mediana` é tão importante?

1. **Captura comportamento esperado por modalidade**
   - Diferentes modalidades de compra têm diferentes valores típicos
   - Feature normaliza valor de cada licitação pela sua modalidade

2. **Identifica desvios significativos**
   - Logaritmo suaviza escala (evita domínio de valores grandes)
   - Ratio mostra quantas vezes acima/abaixo da mediana

3. **Correlação forte com fracionamento**
   - Valores artificialmente baixos (fracionamento) ficarão com log_razao negativo
   - Modelo consegue separa essas estratégias ilícitas

4. **Complementa features estruturais**
   - Baseline tem n_participantes, HHI, concentração
   - Feature adicional (valor) reduz ambiguidade em casos borderline

---

## Recomendações

### 1. **Manter feature no modelo de produção**
- Impacto é decisivo (especialmente em PR-AUC)
- Sem ela, modelo não é viável operacionalmente

### 2. **Usar PR-AUC como métrica principal**
- Para dados desbalanceados, ROC-AUC pode ser enganoso
- PR-AUC reflete melhor a utilidade prática (1,118% melhoria vs 18% em ROC-AUC)

### 3. **Definir threshold de decisão empiricamente**
- Baseline teria que aceitar ~29% FPR para capturar 77% de anomalias
- Com feature, pode usar ~2% FPR para capturar 91% de anomalias

### 4. **Investigar feature importance**
- Executar análise de importância no modelo final
- Confirmar que `log_razao_valor_mediana` é top 3 features

---

## Arquivos Gerados

```
models/
├── modelo_supervisionado_anomalia_baseline.joblib
├── metricas_supervisionado_anomalia_baseline.json
├── modelo_supervisionado_anomalia_com_log_razao.joblib
├── metricas_supervisionado_anomalia_com_log_razao.json
└── ablacao_log_razao_comparison.json
```

Todos os modelos treinados estão salvos com sufixos diferentes para permitir comparação direta.

---

## Conclusão

O experimento de ablação demonstra que `log_razao_valor_mediana` é uma feature **imprescindível** para a detecção de anomalias em licitações. Sua remoção degrada performance em múltiplas dimensões:

- **Decisivo em precisão:** +1,118% melhoria
- **Crítico em recall balanceado:** +28% em balanced accuracy
- **Transformador em casos práticos:** PR-AUC vai de 40% para 91%

**Recomendação final:** Incluir feature no modelo final e considerar análise de feature importance para validar contribuição relativa.
