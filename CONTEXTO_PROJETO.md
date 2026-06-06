# Contexto do Projeto — Detecção de Anomalias em Licitações Públicas

## Sobre este documento

Este arquivo serve como **briefing completo** para um assistente de IA que vai dar continuidade ao projeto. Leia até o fim antes de sugerir qualquer coisa. O projeto já tem decisões tomadas, estrutura montada e dados parcialmente consolidados — não recomece do zero.

---

## 1. Identificação do projeto

- **Nome:** Detecção de Anomalias em Licitações Públicas Federais
- **Tipo:** Trabalho acadêmico individual da disciplina de Inteligência Artificial
- **Linguagem:** Python 3.12.8
- **Sistema operacional:** Windows 11
- **Gerenciador de ambiente:** `uv` (não pip puro, não conda)
- **Editor:** VS Code com extensão Python

## 2. Cronograma e escopo

### Janela de execução
- **Duração:** 2 dias de trabalho dedicado
- **Filosofia:** "Fazer o melhor possível" — escopo intermediário com qualidade > escopo amplo com superficialidade
- **Princípio orientador:** **arquitetura escalável** — código que funcione com 1 ano agora, mas que possa ser estendido para mais anos, outros datasets ou produção sem refatoração grande

### Escopo de dados
- **Recorte temporal:** 12 meses de 2022 (jan-2022 até dez-2022)
- **Motivação do recorte:** evitar dados sujos / possíveis mudanças de schema entre anos; foco em qualidade > volume
- **Os CSVs de 2023 estão baixados e disponíveis** mas serão deixados de lado nesta versão. A arquitetura deve permitir incluí-los depois mudando apenas um parâmetro de configuração.

### Objetivo acadêmico
O trabalho precisa aplicar técnicas de IA da disciplina:
- KNN
- Métodos baseados em distância
- Regressão linear
- Árvores de decisão
- Redes neurais
- CNN
- NLP
- IA generativa

Em tarefas como: classificação, regressão, reconhecimento de padrões, análise textual, análise de dados reais.

**Projeto escolhido:** pipeline integrado de detecção de anomalias (não-supervisionado) + classificação supervisionada com label proxy. Combina os Projetos 8 e 9 originais discutidos, garantindo train/test split (exigência explícita do edital) e diversidade de algoritmos.

## 3. Edital — exigências obrigatórias atendidas

| Seção exigida pelo edital | Como será atendida no projeto |
|---|---|
| Exploração inicial dos dados | Notebook `02_eda.ipynb` (distribuição log, boxplots por modalidade, top fornecedores, sazonalidade, HHI) + inspeção de schemas |
| Pré-processamento incluindo train/test split | Tratamento de tipos, deduplicação, engenharia de features, split estratificado para etapa supervisionada |
| Implementação dos modelos | Isolation Forest + LOF (anomalia, não-supervisionado) e KNN + Árvore de Decisão + Random Forest (supervisionado) |
| Avaliação dos resultados | F1, ROC-AUC, PR-AUC, matriz de confusão para supervisionado; cruzamento com label proxy + inspeção manual + PCA 2D para anomalia |
| Organização e explicação do código | Estrutura `src/` + `notebooks/` + docstrings + README + relatório textual |

### Critérios pontuados (total 10,0)

| Critério | Pontos | Estratégia |
|---|---|---|
| Funcionamento da solução | 2,5 | Pipeline modular, testes rápidos em cada etapa |
| Aplicação correta dos algoritmos | 2,0 | Mix de famílias: anomalia + classificação supervisionada |
| Exploração e tratamento dos dados | 1,5 | EDA forte, tratamento de tipos rigoroso, decisões documentadas |
| Organização e explicação do código | 1,5 | Estrutura `src/` exemplar, docstrings, README, separação notebooks vs código |
| Interpretação dos resultados | 1,0 | Discussão honesta de proxies, limitações de viés, recomendação de uso |
| Apresentação individual e domínio do projeto | 1,5 | Notebook de apresentação preparado em paralelo ao desenvolvimento |

## 4. Stack técnico

### Ambiente virtual
- Localização: `.venv/` na raiz do projeto
- Criado com: `uv venv`
- Ativação: `.venv\Scripts\Activate.ps1` (PowerShell no Windows)
- **Observação:** o prompt PowerShell mostra `(predicao)` em vez de `(.venv)` por causa de resquício antigo do conda no `$PROFILE`. **Não é problema** — o ambiente correto está ativo, confirmado pelo caminho `C:\Users\Gabriela\Pmonkey\IA\predicao\.venv\Scripts\python.exe`.

### Bibliotecas instaladas
Arquivo `requirements.txt`:
```
# Coleta e manipulação
requests>=2.31
pandas>=2.2          # versão atual: 3.0.3
numpy>=1.26
pyarrow>=15.0
python-dotenv>=1.0

# Visualização
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20

# Modelagem
scikit-learn>=1.4
pyod>=1.1
imbalanced-learn>=0.12
xgboost>=2.0

# Interpretabilidade
shap>=0.45

# Redução de dimensionalidade
umap-learn>=0.5

# Notebook
jupyterlab>=4.1
ipykernel>=6.29
ipywidgets>=8.1
tqdm>=4.66

# Qualidade de código
ruff>=0.4
black>=24.0
```

### Comando de instalação
```powershell
uv pip install -r requirements.txt
```

**Regra mental obrigatória:** sempre `uv pip install`, nunca `pip install` puro. O pip puro vaza para o Python global em `C:\Users\Gabriela\AppData\Local\Programs\Python\Python312\`.

## 5. Estrutura de pastas

```
predicao/
├── .venv/                          # ambiente virtual
├── .env                            # variáveis (NÃO commitar)
├── .env.exemple                    # template (typo conhecido — deveria ser .env.example)
├── .gitignore
├── requirements.txt
├── README.md                       # criar — descrição do projeto
├── RELATORIO.md                    # criar — relatório acadêmico em construção
├── CONTEXTO_PROJETO.md             # ESTE ARQUIVO
├── data/
│   ├── raw/
│   │   ├── 202201_Licitacoes/      # uma pasta por mês, JAN-DEZ 2022 (foco do trabalho)
│   │   │   ├── 202201_Licitacao.csv             # SEM ACENTOS
│   │   │   ├── 202201_ItemLicitacao.csv
│   │   │   ├── 202201_ParticipantesLicitacao.csv
│   │   │   └── 202201_EmpenhosRelacionados.csv
│   │   ├── 202202_Licitacoes/
│   │   ├── ...
│   │   ├── 202212_Licitacoes/      # ÚLTIMO mês incluído nesta versão
│   │   ├── 202301_Licitacoes/      # presentes mas IGNORADOS nesta versão
│   │   └── ...
│   ├── interim/                    # parquets consolidados
│   │   ├── licitacao.parquet       # já gerado: ~109.346 linhas, 18 colunas (12 meses de 2022)
│   │   ├── itemlicitacao.parquet   # AINDA NÃO gerado
│   │   ├── participanteslicitacao.parquet  # AINDA NÃO gerado
│   │   └── empenhosrelacionados.parquet    # AINDA NÃO gerado
│   └── processed/                  # dataset analítico final
│       └── dataset_analitico.parquet  # uma linha por licitação, features prontas
├── notebooks/
│   ├── 01_inspecao.ipynb           # schemas, head, info, nulls
│   ├── 02_eda.ipynb                # análise exploratória + lista de candidatos a anomalia
│   ├── 03_features.ipynb           # engenharia de features + dataset analítico
│   ├── 04_modelagem_anomalia.ipynb # Isolation Forest + LOF
│   ├── 05_modelagem_supervisionada.ipynb  # KNN + Árvore + Random Forest
│   ├── 06_avaliacao.ipynb          # comparação, métricas, interpretação
│   └── 99_apresentacao.ipynb       # narrativa final para apresentação oral
└── src/
    ├── __init__.py                 # vazio, só para tornar src um pacote
    ├── config.py                   # carrega .env, expõe caminhos e JANELA TEMPORAL
    ├── consolidacao.py             # une CSVs mensais em parquets
    ├── tratamento.py               # converte tipos, deduplica, normaliza nomes
    ├── features.py                 # engenharia de features reutilizável
    ├── modelos.py                  # wrappers dos modelos + treino padronizado
    └── avaliacao.py                # métricas e visualizações reutilizáveis
```

**Decisão arquitetural importante para escalabilidade:** todo código que pode ser reusado vai para `src/`. Notebooks só contêm chamadas, visualizações e prosa. Isso permite, no futuro:
- Adicionar 2023 mudando uma constante em `config.py`
- Trocar modelo sem reescrever a pipeline de tratamento
- Reusar `features.py` em outro dataset de licitações (estadual, municipal)

## 6. Decisões arquiteturais para escalabilidade

### Configuração centralizada
`src/config.py` deve expor a **janela temporal** como parâmetro explícito:

```python
# Janela atual: só 2022. Mudar aqui para incluir mais anos.
ANOS_INCLUIDOS = [2022]
MESES_INCLUIDOS = list(range(1, 13))  # todos os meses
```

A função de consolidação filtra os arquivos por essas listas. Para incluir 2023 depois, basta `ANOS_INCLUIDOS = [2022, 2023]`.

### Versionamento de dataset analítico
Salvar o dataset final com versionamento na própria pasta:
```
data/processed/dataset_analitico_v1_2022.parquet
data/processed/dataset_analitico_v2_2022_2023.parquet
```
Não sobrescrever — facilita comparação de resultados depois.

### Modelos parametrizados, não hard-coded
Em vez de `IsolationForest(contamination=0.05)`, expor parâmetros em `src/config.py`:
```python
ANOMALIA_CONTAMINATION = 0.05
ANOMALIA_RANDOM_STATE = 42
```

### Persistência de modelos treinados
Salvar modelos com `joblib` em pasta `models/` versionada. Permite carregar modelo sem retreinar.

### Logs estruturados
Cada notebook pode salvar um pequeno JSON em `logs/` com timestamp, tamanho do dataset, métricas obtidas. Útil para comparar execuções.

## 7. Origem dos dados

**Fonte:** Portal da Transparência / Compras.gov.br
**Recorte:** Licitações federais, janeiro/2022 a dezembro/2022 (12 meses)
**Formato:** CSV mensais, separador `;`, encoding `latin-1`

### Modelo de dados (4 tabelas relacionais)

```
Licitacao.csv  (1 linha por licitação)
   ├── ItemLicitacao.csv             (N itens por licitação)
   ├── ParticipantesLicitacao.csv    (N participantes/propostas por licitação)
   └── EmpenhosRelacionados.csv      (N empenhos por licitação)
```

Chave de ligação: a confirmar via inspeção de schema. Provavelmente `id_licitacao` ou `numero_licitacao`.

### Padrões conhecidos do governo brasileiro
- Valores monetários: formato `R$ 1.234,56` (ponto como separador de milhar, vírgula como decimal)
- Datas: formato `DD/MM/AAAA`
- CNPJs: devem ser tratados como string (zeros à esquerda)
- Encoding: `latin-1` (NÃO utf-8)
- Separador CSV: `;` (ponto-e-vírgula)

## 8. Metodologia

### Pipeline integrado em 2 etapas

```
ETAPA A — Detecção de anomalias (não-supervisionado)
  Algoritmos: Isolation Forest + Local Outlier Factor (LOF)
  Output: score de anomalia por licitação
  Justificativa: capta padrões atípicos sem precisar de label

ETAPA B — Classificação supervisionada
  Label proxy: licitação "suspeita" definida via regra de negócio
  Algoritmos: KNN + Árvore de Decisão + Random Forest
  Features: dataset analítico + score de anomalia da Etapa A como feature
  Output: predição binária + probabilidade
  Justificativa: atende ao train/test split do edital + diversifica algoritmos
```

### Label proxy

Como não vamos usar CEIS/CNEP nesta versão (evitar gargalo de API), a label será derivada das próprias variáveis. **Definição preliminar a confirmar após EDA:**

```python
df['suspeita'] = (
    (df['n_participantes'] <= 1) |                      # competição mínima
    (df['valor'] > 3 * df['mediana_modalidade'])        # valor atípico
).astype(int)
```

**No relatório, ser EXPLÍCITO que isso é proxy, não rótulo verdadeiro.** Honestidade metodológica é critério de nota.

### Métricas

- **Para Etapa A:** % de "candidatos visuais" da EDA recuperados nos top-N anômalos, distribuição dos scores, visualização PCA 2D
- **Para Etapa B:** F1, precisão, recall, ROC-AUC, PR-AUC, matriz de confusão (NÃO accuracy — classe desbalanceada)

### Validação cruzada
StratifiedKFold com k=5 na etapa supervisionada. Importante por causa do desbalanceamento.

## 9. Cronograma de 2 dias

### Dia 1 — Dados e EDA

**Manhã (4h)**
- Re-rodar `consolidacao.py` para gerar os 4 parquets completos
- `notebooks/01_inspecao.ipynb`: schemas reais, chave de ligação, nulls, duplicatas
- `src/tratamento.py`: funções de conversão (valores, datas, CNPJs)
- Aplicar tratamento e salvar versões limpas em `data/interim/`

**Tarde (4h)**
- `notebooks/02_eda.ipynb`: distribuição log, boxplots por modalidade, top fornecedores, HHI, sazonalidade
- Lista de candidatos a anomalia identificados visualmente (servirá como referência semi-supervisionada)
- Iniciar `RELATORIO.md` com seções 1 (problema) e 2 (dados)

**Noite (opcional, 2h)**
- `src/features.py`: agregações por licitação (n_itens, dispersão de propostas, etc.)
- `notebooks/03_features.ipynb`: gerar `dataset_analitico.parquet`
- Definir label proxy com base nos achados da EDA

### Dia 2 — Modelagem e entrega

**Manhã (4h)**
- `notebooks/04_modelagem_anomalia.ipynb`: Isolation Forest + LOF
- Comparar com lista de candidatos visuais do dia anterior
- Adicionar score de anomalia ao dataset analítico

**Tarde (4h)**
- `notebooks/05_modelagem_supervisionada.ipynb`: train/test split, KNN, Árvore, Random Forest
- `notebooks/06_avaliacao.ipynb`: métricas, visualizações, comparação
- Continuar `RELATORIO.md` com seções 3 (metodologia) e 4 (resultados)

**Noite (3h)**
- `notebooks/99_apresentacao.ipynb`: narrativa enxuta para apresentação oral
- Fechar `RELATORIO.md` com seção 5 (limitações e trabalhos futuros)
- Polir `README.md`
- Empacotar entrega

### Buffer de risco
Reservar 1-2h em cada dia para imprevistos (encoding, RAM, schema diferente do esperado). Se sobrar, melhorar visualizações ou avançar trecho do dia seguinte.

## 10. O que já foi feito

### ✅ Concluído
1. Ambiente virtual `.venv` criado e funcionando com `uv`
2. Bibliotecas instaladas via `requirements.txt`
3. Estrutura de pastas montada
4. CSVs baixados e organizados em `data/raw/AAAAMM_Licitacoes/`
5. Acentos removidos dos nomes de arquivos via script PowerShell
6. `src/config.py` criado com `load_dotenv()` e caminhos
7. `src/consolidacao.py` criado e EXECUTADO com sucesso para a tabela `Licitacao` (109.346 linhas, 18 colunas — só 2022)
8. `pyarrow` instalado para suporte a parquet

### 🚧 Próximo passo imediato
**Re-rodar `python src/consolidacao.py`** para gerar os 4 parquets em `data/interim/` (Licitacao já rodou; faltam ItemLicitacao, ParticipantesLicitacao e EmpenhosRelacionados — a execução parou no erro de pyarrow antes de processar essas três).

## 11. Decisões metodológicas já tomadas

### Sobre o problema
- **Anomalia estatística ≠ ilegalidade.** Output do modelo é "indício para investigação", nunca "irregularidade comprovada".
- **Viés de seleção** em label proxy: regra heurística não é verdade absoluta. Mencionar como limitação no relatório.

### Sobre o pipeline
- Parquet em vez de CSV para dados intermediários (5x menor, 10x mais rápido).
- `dtype=str` na leitura inicial dos CSVs — conversão de tipos vem depois, com controle explícito.
- Grão da análise: **uma linha por licitação**, com features agregadas das tabelas filhas.
- **12 meses de 2022** (não 24, não 6). Equilíbrio entre representatividade e velocidade.

### Sobre modelagem
- **2 famílias de algoritmos**: detecção de anomalia (IF, LOF) + classificação supervisionada (KNN, árvore, RF).
- Score de anomalia da Etapa A vira feature da Etapa B.
- Validação semi-supervisionada via comparação com candidatos visuais da EDA.

### Sobre features categóricas
- Variáveis de alta cardinalidade (fornecedor, órgão) **não vão para one-hot**. Usar target encoding, frequency encoding, ou agregar em features numéricas. One-hot quebra modelos baseados em distância.

### Cortes feitos para caber em 2 dias
- ❌ Cruzamento com CEIS/CNEP via API (substituído por label proxy)
- ❌ DBSCAN (IF + LOF bastam para Etapa A)
- ❌ SHAP (substituído por `feature_importances_` da árvore)
- ❌ Streamlit dashboard (notebook polido é suficiente)
- ❌ UMAP (PCA cobre a visualização 2D)
- ❌ XGBoost (Random Forest cobre o papel)
- ❌ Dados de 2023 (deixados como capacidade futura — arquitetura permite incluir)

## 12. Estado atual dos arquivos chave

### `src/config.py` (precisa ser atualizado para incluir janela temporal e hiperparâmetros)
Versão alvo:
```python
"""Configurações centralizadas do projeto."""
from dotenv import load_dotenv
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Caminhos
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Credenciais (opcional, não usado nesta versão)
PORTAL_TRANSPARENCIA_API_KEY = os.getenv("PORTAL_TRANSPARENCIA_API_KEY")

# Janela temporal — controla quais arquivos serão processados
ANOS_INCLUIDOS = [2022]
MESES_INCLUIDOS = list(range(1, 13))

# Hiperparâmetros de modelagem
ANOMALIA_CONTAMINATION = 0.05
ANOMALIA_RANDOM_STATE = 42
SUPERVISIONADO_TEST_SIZE = 0.2
SUPERVISIONADO_RANDOM_STATE = 42
```

### `src/consolidacao.py` (precisa ser atualizado para filtrar pela janela temporal)
Versão executada com sucesso para `Licitacao`:
```python
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ENCODING = "latin-1"
SEPARATOR = ";"
TIPOS = ["Licitacao", "ItemLicitacao", "ParticipantesLicitacao", "EmpenhosRelacionados"]


def consolidar_tipo(raw_dir: Path, tipo: str) -> pd.DataFrame:
    arquivos = sorted(raw_dir.glob(f"*/*_{tipo}.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo {tipo} encontrado em {raw_dir}")
    
    print(f"Consolidando {len(arquivos)} arquivos de {tipo}...")
    
    dfs = []
    for arq in tqdm(arquivos):
        ano_mes = arq.stem.split("_")[0]
        df = pd.read_csv(
            arq, sep=SEPARATOR, encoding=ENCODING,
            low_memory=False, dtype=str,
        )
        df["ano_mes_arquivo"] = ano_mes
        dfs.append(df)
    
    consolidado = pd.concat(dfs, ignore_index=True)
    print(f"  → {tipo}: {len(consolidado):,} linhas, {len(consolidado.columns)} colunas")
    return consolidado


def consolidar_tudo(raw_dir: Path, interim_dir: Path):
    interim_dir.mkdir(parents=True, exist_ok=True)
    for tipo in TIPOS:
        df = consolidar_tipo(raw_dir, tipo)
        caminho_saida = interim_dir / f"{tipo.lower()}.parquet"
        df.to_parquet(caminho_saida, index=False)
        print(f"  → Salvo em {caminho_saida}\n")


if __name__ == "__main__":
    consolidar_tudo(
        raw_dir=Path("data/raw"),
        interim_dir=Path("data/interim"),
    )
```

**Refator pendente:** o glob atual `*/*_{tipo}.csv` pega TODOS os meses disponíveis, sem filtrar por `ANOS_INCLUIDOS`. Para escalabilidade real, ajustar para algo como:
```python
arquivos = []
for ano in ANOS_INCLUIDOS:
    for mes in MESES_INCLUIDOS:
        ano_mes = f"{ano}{mes:02d}"
        arquivos.extend(raw_dir.glob(f"{ano_mes}_*/{ano_mes}_{tipo}.csv"))
```

## 13. Problemas resolvidos e armadilhas evitadas

| Problema | Causa | Solução adotada |
|---|---|---|
| `ModuleNotFoundError: pandas` | `(predicao)` do conda ativo, pacotes no Python global | Apagou `.venv`, recriou com `uv venv`, instalou via `uv pip install` |
| Acentos em nomes de arquivo | Governo entrega com `Licitação.csv` (ç e ã) | Script PowerShell para renomear todos sem acento |
| `ImportError: pyarrow required` | Falta no `requirements.txt` | `uv pip install pyarrow` |
| Conda auto-ativando | Instalação antiga | Resquício cosmético no prompt — não funcional |
| Encoding de CSV | UTF-8 não funciona em dados do governo | Usar `encoding="latin-1"` sempre |

## 14. Perfil da pessoa que está fazendo o trabalho

- Estudante de IA com formação em Django/Python
- Trabalha na Defensoria Pública do Estado de Roraima, no DTIC
- Familiar com SQL, Python, Linux (Manjaro pessoal, Ubuntu no trabalho), mas usa Windows neste projeto
- **Prefere explicações detalhadas com contexto** antes do código
- Faz perguntas críticas e ouve bem feedback ("isso é normal?", "preciso disso mesmo?")
- Não tem medo de refazer/reorganizar quando justificado
- **Estilo de resposta preferido:** prosa explicativa + código pontual, com armadilhas marcadas e justificativas das decisões

## 15. Estilo de resposta ideal para a continuidade

- Português brasileiro
- Tom técnico mas didático
- Antes de sugerir código, explicar **por que** está sugerindo aquilo
- Sempre apontar armadilhas conhecidas com a ferramenta/biblioteca
- Quando a usuária cola um erro, **primeiro diagnosticar** (analisar o output completo), **depois propor solução** — não chutar
- Não recomeçar do zero quando a parte anterior já está pronta
- Usar checklists e numeração para passos sequenciais
- Mencionar comandos PowerShell (Windows) e não bash
- Evitar excesso de bullets/headers desnecessários — preferir prosa estruturada
- **Sob pressão de tempo de 2 dias:** priorizar caminho rápido fim-a-fim antes de aprofundar. Funcional > perfeito.

## 16. Princípios de escalabilidade que devem ser respeitados

A solução precisa ser estendível depois sem retrabalho. Diretrizes:

1. **Configuração centralizada** em `src/config.py` — nunca hard-code anos, meses, hiperparâmetros.
2. **Funções puras** em `src/` — receber input, retornar output, sem efeitos colaterais.
3. **Notebooks finos** — só importam de `src/`, visualizam, e narram. Lógica vai para `src/`.
4. **Versionamento de outputs** — dataset analítico salvo com versão no nome.
5. **Persistência de modelos** — salvar com `joblib` para evitar retreino.
6. **Documentação inline** — docstrings em funções de `src/`. Comentários nos notebooks explicando "por quê", não "o quê".
7. **Reprodutibilidade** — sempre `random_state` fixo. Sempre.

---

## Comando para continuidade imediata

```powershell
# 1. Ativar ambiente
.venv\Scripts\Activate.ps1

# 2. Confirmar Python correto
python -c "import sys; print(sys.executable)"
# Esperado: ...\predicao\.venv\Scripts\python.exe

# 3. Re-rodar consolidação completa (agora com pyarrow)
python src/consolidacao.py

# 4. Verificar saídas
dir data\interim
# Esperado: 4 arquivos .parquet (licitacao, itemlicitacao, participanteslicitacao, empenhosrelacionados)
```

Próxima ação após a consolidação: abrir `notebooks/01_inspecao.ipynb` e rodar `df.info()` + `df.head()` em cada parquet para descobrir o schema real e a chave de ligação entre as tabelas.
