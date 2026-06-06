"""Pontos de entrada simples para tarefas do projeto."""

from src.config import PORTAL_TRANSPARENCIA_API_KEY, RAW_DIR
from src.consolidacao import consolidar_tudo


def main() -> None:
	"""Executa a consolidação completa dos CSVs brutos."""
	consolidar_tudo(raw_dir=RAW_DIR)


if __name__ == "__main__":
	main()

