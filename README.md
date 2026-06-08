# Detecção de Anomalias em Licitações Públicas Federais

Pipeline de machine learning para identificação de padrões suspeitos em licitações públicas federais brasileiras, combinando detecção não-supervisionada (Isolation Forest) e classificação supervisionada (Random Forest) sobre dados abertos do Portal da Transparência.

> Trabalho acadêmico — Disciplina de Inteligência Artificial · Python 3.12

---

## Resultados

| Métrica | Valor |
|:---|:---:|
| ROC-AUC | **0,9938** |
| PR-AUC | **0,9065** |
| F1-Score | 0,4766 |
| Recall | 0,9064 |
| Precision | 0,3233 |
| Balanced Accuracy | 0,9426 |

Conjunto de teste estratificado: 39.540 licitações (438 suspeitas · 1,11%).

---

## Metodologia

### Dados

| Tabela | Período | Registros |
|:---|:---:|---:|
| Licitacao | Jan/2022 – Dez/2023 | 197.696 |
| ItemLicitacao | Jan/2022 – Dez/2023 | 1.853.921 |
| ParticipantesLicitacao | Jan/2022 – Dez/2023 | 8.538.480 |
| EmpenhosRelacionados | Jan/2022 – Dez/2023 | 3.166 |

Fonte: [Portal da Transparência · Compras.gov.br](https://portaldatransparencia.gov.br/download-de-dados/licitacoes) — dados públicos, domínio público.

### Pipeline

```
Etapa 1 — Consolidação      CSVs mensais (latin-1, sep=;) → Parquets
Etapa 2 — Tratamento        Tipos, deduplicação, validação de chave composta
Etapa 3 — Features          Agregações (HHI, n_participantes, top1_share) + engenharia
Etapa 4 — Isolation Forest  Detecção não-supervisionada, split 80/20 estratificado
Etapa 5 — Random Forest     Classificação supervisionada com label honesta
```

### Label Honesta

A label de treino é derivada de **regras de negócio independentes do modelo**, sem circularidade com as features:

- **criterio\_3 — Fracionamento**: valor entre R\$17.500 e R\$17.700, faixa que sinaliza fracionar contratos para permanecer abaixo do limite de Dispensa por Valor da Lei 14.133/2021 (R\$17.600 em 2022).
- **criterio\_4 — Alto valor sem competição**: Dispensa ou Inexigibilidade de Licitação com valor acima de R\$10 milhões.

> **Nota metodológica:** a label é heurística, não rótulo verificado de irregularidade. O modelo produz uma lista de priorização para auditoria — anomalia estatística ≠ ilegalidade comprovada.

### Feature Principal: `log_razao_valor_mediana`

```
log_razao_valor_mediana = log₁₀(valor_licitacao / mediana_da_modalidade)
```

Captura o posicionamento relativo do valor dentro da modalidade. Duas licitações com R\$50.000 têm razões opostas se uma é Pregão (mediana ~R\$160k) e outra é Dispensa (mediana ~R\$3,3k). A análise de ablação confirmou que essa feature responde por **56% da importância do modelo** e é responsável por +130% no PR-AUC em relação ao baseline sem ela.

| Configuração | ROC-AUC | PR-AUC | F1 |
|:---|:---:|:---:|:---:|
| Sem `log_razao_valor_mediana` | 0,8398 | 0,3954 | 0,0547 |
| Com `log_razao_valor_mediana` | **0,9938** | **0,9065** | **0,4766** |

### Algoritmos Aplicados

| Algoritmo | Tipo | Papel no projeto |
|:---|:---|:---|
| Isolation Forest | Não-supervisionado | Detecta licitações atípicas sem label; gera `anomalia_score` |
| KNN | Supervisionado | Baseline por proximidade no espaço de features |
| Árvore de Decisão | Supervisionado | Interpretabilidade: extrai regras visuais explícitas |
| **Random Forest** | **Supervisionado** | **Modelo final**: ensemble com `class_weight=balanced` |

---

## Estrutura do Projeto

```
predicao/
├── data/
│   ├── raw/                        # CSVs mensais do Portal da Transparência
│   └── interim/                    # Parquets consolidados e dataset analítico
├── models/
│   ├── modelo_supervisionado_anomalia.joblib
│   └── metricas_supervisionado_anomalia.json
├── notebooks/
│   ├── 02_eda.ipynb                # Análise exploratória dos dados
│   ├── 05b_modelagem_independente.ipynb   # Classificação supervisionada
│   └── 05c_analise_ablacao_knn.ipynb      # Ablação e KNN melhorado
├── src/
│   ├── config.py                   # Configurações centralizadas
│   ├── consolidacao.py             # CSVs → Parquets
│   ├── tratamento.py               # Limpeza e tipagem
│   ├── features.py                 # Feature engineering
│   ├── anomaly.py                  # Isolation Forest com train/test split
│   └── run_pipeline.py             # Orquestrador end-to-end
├── logs/                           # Logs de execução por data/hora
├── docs/                           # Documentação técnica adicional
└── requirements.txt
```

---

## Como Reproduzir

### Pré-requisitos

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) para gerenciamento de ambiente

### Instalação

```powershell
git clone <repo>
cd predicao

uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### Dados

Baixar os CSVs mensais de [licitações do Portal da Transparência](https://portaldatransparencia.gov.br/download-de-dados/licitacoes) para `data/raw/`. Cada mês gera uma pasta `AAAAMM_Licitacoes/` com 4 arquivos CSV (Licitacao, ItemLicitacao, ParticipantesLicitacao, EmpenhosRelacionados).

A janela temporal é controlada em `src/config.py`:

```python
ANOS_INCLUIDOS = [2022, 2023]
MESES_INCLUIDOS = list(range(1, 13))
```

### Execução

```powershell
python -X utf8 -m src.run_pipeline
```

Tempo estimado: 8–12 minutos (depende do hardware). Saídas geradas:

| Arquivo | Descrição |
|:---|:---|
| `data/interim/dataset_analitico_com_scores.parquet` | Dataset com scores do Isolation Forest |
| `models/modelo_supervisionado_anomalia.joblib` | Modelo treinado serializável |
| `models/metricas_supervisionado_anomalia.json` | Métricas no conjunto de teste |
| `logs/consolidacao_*.json` | Metadata de cada execução de consolidação |

---

## Limitações Metodológicas

1. **Label proxy**: criterio\_3 e criterio\_4 são heurísticas legais, não confirmações de irregularidade. Alto valor em Dispensa pode ser legítimo (ex.: medicamento de fornecedor único mundial).
2. **Ausência de ground truth**: sem cruzamento com CEIS, CNEP ou processos administrativos, não é possível estimar a taxa real de verdadeiros positivos.
3. **Viés temporal**: modelo treinado em 2022–2023. Mudanças no teto legal de dispensa ou no perfil de contratações podem afetar a generalização para anos futuros.
4. **Granularidade**: análise no nível da licitação. Esquemas de fracionamento que operam entre licitações (mesmo fornecedor, licitações separadas) não são capturados por este modelo.
