"""
Configurações centralizadas do projeto.

Este é o ÚNICO lugar onde caminhos, parâmetros e credenciais devem ser
declarados. Qualquer outro módulo ou notebook que precise dessas informações
deve importar daqui.

Princípio: para mudar comportamento global do projeto (incluir mais anos,
trocar contamination do Isolation Forest, etc.), basta editar este arquivo.
"""
from dotenv import load_dotenv
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# ROOT_DIR é a raiz do projeto (uma pasta acima de src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Carrega o .env da raiz do projeto, independente de onde for chamado
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Credenciais (opcionais — não usadas nesta versão do projeto)
# ---------------------------------------------------------------------------
PORTAL_TRANSPARENCIA_API_KEY = os.getenv("PORTAL_TRANSPARENCIA_API_KEY")

# ---------------------------------------------------------------------------
# Janela temporal
# ---------------------------------------------------------------------------
# Controla quais meses serão processados na consolidação.
# Para incluir 2023 no futuro: ANOS_INCLUIDOS = [2022, 2023]
ANOS_INCLUIDOS = [2022, 2023]
MESES_INCLUIDOS = list(range(1, 13))  # janeiro a dezembro

# Chave composta usada em todo o projeto para identificar uma licitação
# Mantemos aqui para centralizar a convenção e evitar duplicação entre
# `src/tratamento.py`, notebooks e outros módulos.
CHAVE_LICITACAO = ["numero_licitacao", "codigo_ug", "codigo_modalidade_compra"]

# ---------------------------------------------------------------------------
# Formato dos CSVs do governo brasileiro
# ---------------------------------------------------------------------------
# Dados do Portal da Transparência / Compras.gov.br seguem este padrão:
CSV_ENCODING = "latin-1"  # NÃO é utf-8
CSV_SEPARATOR = ";"       # NÃO é vírgula

# Tabelas que compõem o modelo relacional de licitações:
#   Licitacao (1) ─┬─ ItemLicitacao (N)
#                  ├─ ParticipantesLicitacao (N)
#                  └─ EmpenhosRelacionados (N)
TIPOS_TABELAS = [
    "Licitacao",
    "ItemLicitacao",
    "ParticipantesLicitacao",
    "EmpenhosRelacionados",
]

# ---------------------------------------------------------------------------
# Hiperparâmetros de modelagem
# ---------------------------------------------------------------------------
# Detecção de anomalias (Etapa A — não-supervisionado)
ANOMALIA_CONTAMINATION = 0.05   # estimativa de % de anomalias no dataset
ANOMALIA_RANDOM_STATE = 42

# Classificação supervisionada (Etapa B)
SUPERVISIONADO_TEST_SIZE = 0.2
SUPERVISIONADO_RANDOM_STATE = 42
SUPERVISIONADO_CV_FOLDS = 5     # k para StratifiedKFold

# ---------------------------------------------------------------------------
# Versão do dataset (para versionamento de outputs)
# ---------------------------------------------------------------------------
# Atualizar quando mudar a janela temporal ou metodologia de features
DATASET_VERSION = "v1_2022"