import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configurações compartilhadas
ENCODING = "latin-1"
SEPARATOR = ";"

TIPOS = ["Licitacao", "ItemLicitacao", "ParticipantesLicitacao", "EmpenhosRelacionados"]


def consolidar_tipo(raw_dir: Path, tipo: str) -> pd.DataFrame:
    """
    Lê todos os arquivos de um tipo (ex.: 'Licitacao') de todos os meses
    e retorna um DataFrame único.
    """
    arquivos = sorted(raw_dir.glob(f"*/*_{tipo}.csv"))
    
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo {tipo} encontrado em {raw_dir}")
    
    print(f"Consolidando {len(arquivos)} arquivos de {tipo}...")
    
    dfs = []
    for arq in tqdm(arquivos):
        # Extrai o mês do nome do arquivo (ex.: '202305')
        ano_mes = arq.stem.split("_")[0]
        
        df = pd.read_csv(
            arq,
            sep=SEPARATOR,
            encoding=ENCODING,
            low_memory=False,  # evita warning de tipos mistos
            dtype=str,         # lê tudo como string primeiro, converte depois
        )
        df["ano_mes_arquivo"] = ano_mes  # rastreio de origem
        dfs.append(df)
    
    consolidado = pd.concat(dfs, ignore_index=True)
    print(f"  → {tipo}: {len(consolidado):,} linhas, {len(consolidado.columns)} colunas")
    return consolidado


def consolidar_tudo(raw_dir: Path, interim_dir: Path):
    """Roda a consolidação para todos os tipos e salva em parquet."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    for tipo in TIPOS:
        df = consolidar_tipo(raw_dir, tipo)
        # Parquet é muito mais rápido pra reabrir e ocupa 5-10x menos espaço
        caminho_saida = interim_dir / f"{tipo.lower()}.parquet"
        df.to_parquet(caminho_saida, index=False)
        print(f"  → Salvo em {caminho_saida}\n")


if __name__ == "__main__":
    consolidar_tudo(
        raw_dir=Path("data/raw"),
        interim_dir=Path("data/interim"),
    )