"""
Consolidação de CSVs mensais em arquivos parquet únicos.

Lê todos os CSVs dos meses configurados em config.ANOS_INCLUIDOS e
config.MESES_INCLUIDOS, e consolida cada tipo de tabela em um único
parquet em data/interim/.

Para incluir novos anos/meses, edite as constantes em src/config.py
e re-execute este script. Nenhuma alteração de código é necessária aqui.

Uso:
    python src/consolidacao.py

Ou de dentro de um notebook:
    from src.consolidacao import consolidar_tudo
    metadata = consolidar_tudo()
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Permite executar tanto como script (`python src/consolidacao.py`) quanto
# como módulo (`python -m src.consolidacao`) ou import de notebook.
# Adiciona a raiz do projeto ao sys.path para que `from src.config import ...`
# funcione independente do diretório de execução.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402 — import depende do sys.path acima
    ANOS_INCLUIDOS,
    CSV_ENCODING,
    CSV_SEPARATOR,
    INTERIM_DIR,
    LOGS_DIR,
    MESES_INCLUIDOS,
    RAW_DIR,
    TIPOS_TABELAS,
)


def listar_arquivos_esperados(
    raw_dir: Path,
    tipo: str,
    anos: List[int],
    meses: List[int],
) -> List[Path]:
    """
    Lista os caminhos dos CSVs de um tipo de tabela respeitando a janela
    temporal configurada.

    Procura por arquivos no padrão `{ano_mes}_*/{ano_mes}_{tipo}.csv`,
    onde `ano_mes` é a concatenação de ano (4 dígitos) e mês (2 dígitos).
    Por exemplo, para `tipo='Licitacao'`, ano=2022, mês=3, procura:
        data/raw/202203_*/202203_Licitacao.csv

    Parâmetros
    ----------
    raw_dir : Path
        Diretório raiz dos dados brutos (ex.: `data/raw/`)
    tipo : str
        Nome do tipo de tabela (ex.: 'Licitacao', 'ItemLicitacao')
    anos : List[int]
        Anos a incluir (ex.: `[2022]` ou `[2022, 2023]`)
    meses : List[int]
        Meses a incluir, valores de 1 a 12

    Retorna
    -------
    List[Path]
        Lista de caminhos encontrados, em ordem cronológica.
        Meses configurados mas sem arquivo correspondente em disco
        geram aviso no terminal, mas não interrompem a execução.
    """
    encontrados: List[Path] = []
    faltantes: List[str] = []

    for ano in sorted(anos):
        for mes in sorted(meses):
            ano_mes = f"{ano}{mes:02d}"
            padrao = f"{ano_mes}_*/{ano_mes}_{tipo}.csv"
            candidatos = list(raw_dir.glob(padrao))

            if candidatos:
                encontrados.extend(candidatos)
            else:
                faltantes.append(ano_mes)

    if faltantes:
        print(f"  ⚠️  {tipo}: meses sem arquivo em disco: {', '.join(faltantes)}")

    return encontrados


def consolidar_tipo(
    arquivos: List[Path],
    tipo: str,
    encoding: str = CSV_ENCODING,
    separator: str = CSV_SEPARATOR,
) -> pd.DataFrame:
    """
    Lê e consolida uma lista de CSVs do mesmo tipo em um único DataFrame.

    Estratégia: tudo é lido como string (`dtype=str`). A conversão para
    tipos corretos (float, datetime, etc.) é responsabilidade da etapa
    seguinte (`src/tratamento.py`). Isso evita inferências silenciosas
    do pandas que costumam dar problema com dados brasileiros (vírgula
    decimal, datas DD/MM/AAAA, CNPJs com zero à esquerda).

    Parâmetros
    ----------
    arquivos : List[Path]
        Lista de caminhos de CSVs a consolidar. Todos devem ter o mesmo
        schema (mesmas colunas).
    tipo : str
        Nome do tipo, usado apenas para logging.
    encoding : str, padrão = CSV_ENCODING
        Encoding dos arquivos.
    separator : str, padrão = CSV_SEPARATOR
        Caractere separador de colunas.

    Retorna
    -------
    pd.DataFrame
        DataFrame consolidado com coluna extra `ano_mes_arquivo`,
        útil para rastrear de qual mês veio cada linha (sem isso,
        deduplicação e auditoria ficam difíceis).

    Levanta
    -------
    FileNotFoundError
        Se a lista de arquivos estiver vazia.
    """
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo {tipo} para consolidar.")

    print(f"📂 Consolidando {len(arquivos)} arquivos de {tipo}...")

    dfs: List[pd.DataFrame] = []
    for arq in tqdm(arquivos, desc=tipo, leave=False):
        # Extrai 'AAAAMM' do nome do arquivo (ex.: '202203_Licitacao.csv' -> '202203')
        ano_mes = arq.stem.split("_")[0]

        df = pd.read_csv(
            arq,
            sep=separator,
            encoding=encoding,
            low_memory=False,  # evita warning de tipos mistos
            dtype=str,         # leitura preserva strings; conversão é depois
        )
        df["ano_mes_arquivo"] = ano_mes
        dfs.append(df)

    consolidado = pd.concat(dfs, ignore_index=True)
    print(f"  → {tipo}: {len(consolidado):,} linhas, {len(consolidado.columns)} colunas")
    return consolidado


def salvar_metadata(metadata: Dict, logs_dir: Path) -> Path:
    """
    Salva um JSON com informações sobre esta execução de consolidação.

    Útil para o relatório acadêmico (rastreabilidade de qual versão dos
    dados gerou quais resultados) e para auditoria.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = logs_dir / f"consolidacao_{timestamp}.json"

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"📋 Metadata salva em {caminho}")
    return caminho


def consolidar_tudo(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    anos: List[int] = None,
    meses: List[int] = None,
    tipos: List[str] = None,
    salvar_log: bool = True,
) -> Dict:
    """
    Pipeline completo de consolidação.

    Para cada tipo de tabela configurado, lista os arquivos da janela
    temporal, consolida em DataFrame único e salva como parquet em
    `interim_dir`.

    Os parâmetros opcionais permitem override pontual (útil em testes),
    mas o uso recomendado é configurar tudo em `src/config.py` e chamar
    sem argumentos.

    Parâmetros
    ----------
    raw_dir : Path
        Diretório dos dados brutos.
    interim_dir : Path
        Diretório onde os parquets serão salvos.
    anos : List[int], opcional
        Override de ANOS_INCLUIDOS. Se None, usa o valor do config.
    meses : List[int], opcional
        Override de MESES_INCLUIDOS. Se None, usa o valor do config.
    tipos : List[str], opcional
        Override de TIPOS_TABELAS. Se None, usa o valor do config.
    salvar_log : bool
        Se True, escreve um JSON em LOGS_DIR com metadados da execução.

    Retorna
    -------
    Dict
        Metadata da execução: timestamp, janela temporal, e estatísticas
        por tabela (n_arquivos, n_linhas, n_colunas, caminho do parquet).
    """
    # Aplica defaults aqui (não na assinatura) para evitar mutable default args
    anos = anos if anos is not None else ANOS_INCLUIDOS
    meses = meses if meses is not None else MESES_INCLUIDOS
    tipos = tipos if tipos is not None else TIPOS_TABELAS

    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONSOLIDAÇÃO DE LICITAÇÕES")
    print("=" * 60)
    print(f"📅 Janela: anos={anos}, meses={meses}")
    print(f"📁 Origem: {raw_dir}")
    print(f"📁 Destino: {interim_dir}")
    print()

    metadata: Dict = {
        "timestamp": datetime.now().isoformat(),
        "anos_incluidos": anos,
        "meses_incluidos": meses,
        "tabelas": {},
    }

    for tipo in tipos:
        arquivos = listar_arquivos_esperados(raw_dir, tipo, anos, meses)

        if not arquivos:
            print(f"❌ Nenhum arquivo encontrado para {tipo}. Pulando.\n")
            metadata["tabelas"][tipo] = {"status": "sem_arquivos"}
            continue

        df = consolidar_tipo(arquivos, tipo)

        caminho_saida = interim_dir / f"{tipo.lower()}.parquet"
        df.to_parquet(caminho_saida, index=False)
        print(f"  💾 Salvo em {caminho_saida}\n")

        metadata["tabelas"][tipo] = {
            "status": "sucesso",
            "n_arquivos": len(arquivos),
            "n_linhas": len(df),
            "n_colunas": len(df.columns),
            "arquivo_saida": str(caminho_saida),
        }

    if salvar_log:
        salvar_metadata(metadata, LOGS_DIR)

    print("=" * 60)
    print("✅ Consolidação finalizada.")
    print("=" * 60)
    return metadata


if __name__ == "__main__":
    consolidar_tudo()