"""
Tratamento de dados das tabelas consolidadas.

Aplica conversões de tipo, normalização de nomes de coluna e validações
sobre os parquets gerados pela etapa de consolidação. Salva versões
limpas em data/interim/ com sufixo `_tratado`.

Decisões metodológicas (calibradas após inspeção em 01_inspecao.ipynb):

- data_abertura é preenchida APENAS para modalidade Pregão (30k de 109k);
demais modalidades terão NaT após conversão. Isso é esperado.
- Chave primária composta: (numero_licitacao, codigo_ug, codigo_modalidade_compra)
- Valores monetários no padrão "703363,4400" (sem milhar, vírgula decimal)
- Datas no padrão "DD/MM/AAAA"
- Códigos identificadores (UG, Órgão, Processo) mantidos como STRING
  para preservar zeros à esquerda
- NÃO deduplicar Licitacao — chave composta já garante unicidade
- data_abertura tem ~66% de nulls; conversão é resiliente (NaT onde faltar)

Uso:
    from src.tratamento import tratar_tudo
    tratar_tudo()
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd

# Garante que `from src.config import ...` funcione independente do CWD
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import INTERIM_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Constantes derivadas da inspeção dos dados
# ---------------------------------------------------------------------------

# Chave composta que identifica unicamente uma licitação.
# Confirmado em 01_inspecao.ipynb: 0 duplicatas em 109.346 registros.
CHAVE_LICITACAO = ["numero_licitacao", "codigo_ug", "codigo_modalidade_compra"]

# Colunas que devem ser convertidas para float (valor monetário em padrão BR)
COLUNAS_MONETARIAS = {
    "licitacao": ["valor_licitacao"],
    "itemlicitacao": ["valor_item", "quantidade_item"],
    "participanteslicitacao": [],  # confirmar se há colunas de valor
    "empenhosrelacionados": ["valor_empenho_r"],
}

# Colunas que devem ser convertidas para datetime (formato DD/MM/AAAA)
COLUNAS_DATA = {
    "licitacao": ["data_abertura", "data_resultado_compra"],
    "itemlicitacao": [],
    "participanteslicitacao": [],
    "empenhosrelacionados": ["data_emissao_empenho"],
}


# ---------------------------------------------------------------------------
# Funções utilitárias de conversão
# ---------------------------------------------------------------------------

def normalizar_nome_coluna(nome: str) -> str:
    """
    Converte um nome de coluna para snake_case ASCII.

    Exemplos:
        'Número Licitação'    -> 'numero_licitacao'
        'Valor Empenho (R$)'  -> 'valor_empenho_rs'
        'Código UG'           -> 'codigo_ug'
    """
    # Remove acentos via normalização Unicode
    sem_acento = (
        unicodedata.normalize("NFKD", nome)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # Minúsculas e trim
    s = sem_acento.lower().strip()
    # Remove caracteres especiais comuns
    s = (
        s.replace("(", "")
        .replace(")", "")
        .replace("$", "")
        .replace("/", "_")
        .replace("-", "_")
    )
    # Espaços e múltiplos underscores viram um único underscore
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica normalização snake_case a todas as colunas de um DataFrame.
    Não modifica o DataFrame original (retorna cópia).
    """
    df = df.copy()
    df.columns = [normalizar_nome_coluna(c) for c in df.columns]
    return df


def converter_valor_brasileiro(serie: pd.Series) -> pd.Series:
    """
    Converte série de valores no padrão BR para float.

    Padrão de entrada: "703363,4400" (vírgula como separador decimal,
    sem separador de milhar — confirmado na inspeção dos dados).

    Valores vazios, None ou não-conversíveis viram NaN.
    """
    return (
        serie.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})
        .astype(float)
    )


def converter_data_brasileira(serie: pd.Series) -> pd.Series:
    """
    Converte série de datas no padrão BR (DD/MM/AAAA) para datetime.

    Valores nulos ou em formato inválido viram NaT (sem warning).
    """
    return pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")


# ---------------------------------------------------------------------------
# Validações
# ---------------------------------------------------------------------------

def validar_chave_unica(
    df: pd.DataFrame,
    chave: List[str],
    nome_tabela: str = "tabela",
) -> None:
    """
    Verifica se a chave composta é única na tabela.
    Levanta ValueError se houver duplicatas.

    Útil para auditoria pós-tratamento — se o tratamento (ex.: normalização
    de strings) introduziu duplicatas inesperadas, esta função pega.
    """
    n_dup = df.duplicated(subset=chave).sum()
    if n_dup > 0:
        raise ValueError(
            f"{nome_tabela}: chave {chave} tem {n_dup:,} duplicatas. "
            f"Esperado: 0."
        )
    print(f"  ✓ {nome_tabela}: chave {chave} é única ({len(df):,} registros)")


# ---------------------------------------------------------------------------
# Pipelines de tratamento por tabela
# ---------------------------------------------------------------------------

def tratar_tabela_generica(
    df: pd.DataFrame,
    nome_tabela: str,
    colunas_monetarias: List[str],
    colunas_data: List[str],
) -> pd.DataFrame:
    """
    Pipeline genérico de tratamento aplicável a qualquer uma das 4 tabelas.

    Etapas:
    1. Normalização de nomes de coluna (snake_case ASCII)
    2. Conversão de colunas monetárias (vírgula BR -> float)
    3. Conversão de colunas de data (DD/MM/AAAA -> datetime)
    4. Códigos permanecem como string (preservação de zeros à esquerda)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame consolidado de uma das tabelas.
    nome_tabela : str
        Nome para logging.
    colunas_monetarias : List[str]
        Colunas a converter para float. Nomes já em snake_case.
    colunas_data : List[str]
        Colunas a converter para datetime. Nomes já em snake_case.

    Retorna
    -------
    pd.DataFrame
        DataFrame tratado.
    """
    print(f"\n🔧 Tratando {nome_tabela}...")
    print(f"  • Linhas: {len(df):,}")

    # 1. Normalização de nomes
    df = normalizar_colunas(df)

    # 2. Conversões monetárias
    for col in colunas_monetarias:
        if col in df.columns:
            n_antes_null = df[col].isnull().sum()
            df[col] = converter_valor_brasileiro(df[col])
            n_depois_null = df[col].isnull().sum()
            novos_nulls = n_depois_null - n_antes_null
            if novos_nulls > 0:
                print(
                    f"  ⚠️  {col}: {novos_nulls:,} valores não puderam ser "
                    f"convertidos para float (viraram NaN)"
                )
        else:
            print(f"  ⚠️  Coluna monetária '{col}' não encontrada — ignorando")

    # 3. Conversões de data
    for col in colunas_data:
        if col in df.columns:
            df[col] = converter_data_brasileira(df[col])
        else:
            print(f"  ⚠️  Coluna de data '{col}' não encontrada — ignorando")

    print(f"  ✓ Tratamento concluído")
    return df


def tratar_licitacao(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline específico da tabela Licitacao."""
    df = tratar_tabela_generica(
        df,
        nome_tabela="Licitacao",
        colunas_monetarias=COLUNAS_MONETARIAS["licitacao"],
        colunas_data=COLUNAS_DATA["licitacao"],
    )
    validar_chave_unica(df, CHAVE_LICITACAO, "Licitacao")
    return df


def tratar_participantes(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline específico da tabela ParticipantesLicitacao."""
    df = tratar_tabela_generica(
        df,
        nome_tabela="ParticipantesLicitacao",
        colunas_monetarias=COLUNAS_MONETARIAS["participanteslicitacao"],
        colunas_data=COLUNAS_DATA["participanteslicitacao"],
    )

    # Normalização da flag de vencedor para booleano
    if "flag_vencedor" in df.columns:
        df["flag_vencedor"] = (
            df["flag_vencedor"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map({"SIM": True, "NÃO": False, "NAO": False})
        )
        # Valida que não criou NaN inesperado
        n_nulls = df["flag_vencedor"].isnull().sum()
        if n_nulls > 0:
            print(
                f"  ⚠️  flag_vencedor: {n_nulls:,} valores não foram mapeados "
                f"(viraram NaN). Verifique valores únicos com value_counts()."
            )
    return df


def tratar_itens(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline específico da tabela ItemLicitacao."""
    return tratar_tabela_generica(
        df,
        nome_tabela="ItemLicitacao",
        colunas_monetarias=COLUNAS_MONETARIAS["itemlicitacao"],
        colunas_data=COLUNAS_DATA["itemlicitacao"],
    )


def tratar_empenhos(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline específico da tabela EmpenhosRelacionados."""
    return tratar_tabela_generica(
        df,
        nome_tabela="EmpenhosRelacionados",
        colunas_monetarias=COLUNAS_MONETARIAS["empenhosrelacionados"],
        colunas_data=COLUNAS_DATA["empenhosrelacionados"],
    )


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def tratar_tudo(
    interim_dir: Path = INTERIM_DIR,
    sufixo_saida: str = "_tratado",
) -> None:
    """
    Lê os parquets brutos consolidados, aplica tratamento e salva versões
    limpas no mesmo diretório com sufixo `_tratado`.

    Estrutura de I/O:
        Lê:    data/interim/licitacao.parquet
        Salva: data/interim/licitacao_tratado.parquet
    """
    print("=" * 60)
    print("TRATAMENTO DE DADOS")
    print("=" * 60)

    pipelines = {
        "licitacao": tratar_licitacao,
        "itemlicitacao": tratar_itens,
        "participanteslicitacao": tratar_participantes,
        "empenhosrelacionados": tratar_empenhos,
    }

    for nome, fn in pipelines.items():
        caminho_entrada = interim_dir / f"{nome}.parquet"
        caminho_saida = interim_dir / f"{nome}{sufixo_saida}.parquet"

        if not caminho_entrada.exists():
            print(f"\n⚠️  {caminho_entrada} não encontrado. Pulando.")
            continue

        df = pd.read_parquet(caminho_entrada)
        df_tratado = fn(df)
        df_tratado.to_parquet(caminho_saida, index=False)
        print(f"  💾 Salvo em {caminho_saida}")

    print("\n" + "=" * 60)
    print("✅ Tratamento finalizado.")
    print("=" * 60)


if __name__ == "__main__":
    tratar_tudo()
