"""Configurações centralizadas do projeto."""
from dotenv import load_dotenv
from pathlib import Path
import os

# Carrega o .env da raiz do projeto, independente de onde for chamado
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Caminhos
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Credenciais
PORTAL_TRANSPARENCIA_API_KEY = os.getenv("PORTAL_TRANSPARENCIA_API_KEY")

if not PORTAL_TRANSPARENCIA_API_KEY:
    raise RuntimeError(
        "PORTAL_TRANSPARENCIA_API_KEY não encontrada. "
        "Verifique o arquivo .env na raiz do projeto."
    )