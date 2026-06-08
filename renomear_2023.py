import pandas as pd
from src.config import INTERIM_DIR

# 1. Quantas linhas tem o licitacao_tratado?
df_lic = pd.read_parquet(INTERIM_DIR / "licitacao_tratado.parquet")
print(f"licitacao_tratado: {len(df_lic):,} linhas")

# 2. Distribuição por ano_mes_arquivo
if "ano_mes_arquivo" in df_lic.columns:
    print("\nDistribuição por mês (primeiros 6 e últimos 6):")
    counts = df_lic["ano_mes_arquivo"].value_counts().sort_index()
    print("Primeiros 6:", counts.head(6).to_dict())
    print("Últimos 6:", counts.tail(6).to_dict())
    print(f"\nTotal de meses únicos: {counts.shape[0]}")

# 3. Quantas linhas tem o dataset_analitico?
df_analitico = pd.read_parquet(INTERIM_DIR / "dataset_analitico.parquet")
print(f"\ndataset_analitico: {len(df_analitico):,} linhas")

# 4. Mesma distribuição
if "ano_mes_arquivo" in df_analitico.columns:
    print("\nDistribuição do dataset analítico por mês:")
    counts2 = df_analitico["ano_mes_arquivo"].value_counts().sort_index()
    print("Primeiros 6:", counts2.head(6).to_dict())
    print("Últimos 6:", counts2.tail(6).to_dict())
    print(f"\nTotal de meses únicos: {counts2.shape[0]}")

# 5. Se tem coluna 'ano'
if "ano" in df_analitico.columns:
    print(f"\nDistribuição por ano:")
    print(df_analitico["ano"].value_counts().sort_index())