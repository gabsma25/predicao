from pathlib import Path
import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, CHAVE_LICITACAO


def build_features():
    # Leitura
    lic_path = INTERIM_DIR / "licitacao_tratado.parquet"
    par_path = INTERIM_DIR / "participanteslicitacao_tratado.parquet"
    itens_path = INTERIM_DIR / "itemlicitacao_tratado.parquet"

    df_lic = pd.read_parquet(lic_path)
    df_par = pd.read_parquet(par_path)
    df_itens = pd.read_parquet(itens_path)

    # Contagem de participantes por licitação
    n_part = (
        df_par.groupby(CHAVE_LICITACAO)
        .agg(n_participantes=("codigo_participante", "nunique"))
        .reset_index()
    )

    # Agregados de itens por licitação
    itens_agg = (
        df_itens.groupby(CHAVE_LICITACAO)
        .agg(n_itens=("valor_item", "count"), valor_total_itens=("valor_item", "sum"))
        .reset_index()
    )

    # HHI e participação do maior fornecedor (por valor de itens dentro da licitação)
    supplier = (
        df_itens.groupby(CHAVE_LICITACAO + ["nome_vencedor"])
        .agg(valor_supplier=("valor_item", "sum"))
        .reset_index()
    )

    supplier = supplier.dropna(subset=["valor_supplier"]).copy()
    supplier = supplier.assign(
        valor_total_items=lambda d: d.groupby(CHAVE_LICITACAO)["valor_supplier"].transform("sum")
    )
    supplier = supplier.assign(share=lambda d: d["valor_supplier"] / d["valor_total_items"]) 

    hhi = (
        supplier.assign(share2=lambda d: d["share"] ** 2)
        .groupby(CHAVE_LICITACAO)["share2"]
        .sum()
        .reset_index()
        .rename(columns={"share2": "hhi"})
    )

    top1 = (
        supplier.groupby(CHAVE_LICITACAO)["valor_supplier"]
        .agg(valor_top1="max")
        .reset_index()
    )

    total_items = (
        supplier.groupby(CHAVE_LICITACAO)["valor_supplier"]
        .sum()
        .reset_index()
        .rename(columns={"valor_supplier": "valor_total_items"})
    )

    top1 = top1.merge(total_items, on=CHAVE_LICITACAO, how="left").assign(
        top1_share=lambda d: d["valor_top1"] / d["valor_total_items"]
    )

    # Merge features into licitações
    df = df_lic.merge(n_part, on=CHAVE_LICITACAO, how="left")
    df = df.merge(itens_agg, on=CHAVE_LICITACAO, how="left")
    df = df.merge(hhi, on=CHAVE_LICITACAO, how="left")
    df = df.merge(top1[[*CHAVE_LICITACAO, "top1_share"]], on=CHAVE_LICITACAO, how="left")

    # Preenchimentos e novas features
    df["n_participantes"] = df["n_participantes"].fillna(0).astype(int)
    df["n_itens"] = df["n_itens"].fillna(0).astype(int)
    df["valor_total_itens"] = df.get("valor_total_itens", df.get("valor_total_items", 0)).fillna(0)

    df["log_valor_licitacao"] = np.log10(df["valor_licitacao"].clip(lower=0.01))

    # Razão log10 do valor em relação à mediana da modalidade
    # Captura "quão atípico é esse valor dentro do contexto da modalidade" —
    # duas licitações com mesmo valor absoluto terão log_razao diferente se
    # pertencerem a modalidades com medianas distintas.
    df["valor_mediana_modalidade"] = df.groupby("modalidade_compra")["valor_licitacao"].transform("median")
    df["log_razao_valor_mediana"] = np.log10(
        df["valor_licitacao"].clip(lower=0.01) / df["valor_mediana_modalidade"].clip(lower=0.01)
    )

    # Tempo
    if "data_resultado_compra" in df.columns:
        df["mes"] = df["data_resultado_compra"].dt.to_period("M").astype(str)
        df["ano"] = df["data_resultado_compra"].dt.year
        df["dia_semana"] = df["data_resultado_compra"].dt.dayofweek

    # Critérios heurísticos (para referência/comparação apenas)
    # Não são usados como label no modelo supervisionado para evitar circularidade
    df["valor_p99_modalidade"] = df.groupby("modalidade_compra")["valor_licitacao"].transform(lambda s: s.quantile(0.99))
    df["criterio_1_acima_p99_modalidade"] = df["valor_licitacao"] > df["valor_p99_modalidade"]

    df["criterio_2_sem_competicao"] = df["n_participantes"] == 1

    df["criterio_3_limite_dispensa"] = (df["valor_licitacao"] >= 17500) & (df["valor_licitacao"] <= 17700)

    modalidades_sem_comp = ["Dispensa de Licitação", "Inexigibilidade de Licitação"]
    df["criterio_4_alto_valor_sem_comp"] = (
        df["modalidade_compra"].isin(modalidades_sem_comp) & (df["valor_licitacao"] > 10_000_000)
    )

    # Candidato anomalia: usado apenas para validação/comparação, não para treinar modelo
    df["candidato_anomalia_heuristica"] = (
        df["criterio_1_acima_p99_modalidade"] |
        df["criterio_2_sem_competicao"] |
        df["criterio_3_limite_dispensa"] |
        df["criterio_4_alto_valor_sem_comp"]
    )

    # Salvamento
    out_path = INTERIM_DIR / "dataset_analitico.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Dataset analítico salvo em: {out_path}")
    print(f"Linhas: {len(df):,}, colunas: {len(df.columns)}")

    return out_path


def main():
    build_features()


if __name__ == "__main__":
    main()
