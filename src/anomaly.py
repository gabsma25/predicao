import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DIR, CHAVE_LICITACAO


def run_isolation_forest_with_split(
    test_size: float = 0.2,
    contamination: float = 0.01,
    random_state: int = 42
):
    """
    Treina Isolation Forest com split treino/teste.
    
    IMPORTANTE: Treinamos no conjunto de treino e avaliamos em ambos
    (aplicando scores globais para comparação com supervisionado).
    
    Retorna dataset com colunas:
    - anomalia_score: score contínuo do IF (maior = mais anômalo)
    - anomalia_label: previsão binária do IF (True = anomalia)
    - fold: 'train' ou 'test' (para downstream fazer split correto)
    """
    path = INTERIM_DIR / "dataset_analitico.parquet"
    df = pd.read_parquet(path)

    # Features para treino
    feats = [
        "log_valor_licitacao",
        "n_participantes",
        "n_itens",
        "hhi",
        "top1_share",
        "valor_total_itens",
    ]

    for c in feats:
        if c not in df.columns:
            df[c] = 0

    X = df[feats].fillna(0).astype(float)

    # Split treino/teste
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df.groupby("modalidade_compra").ngroup()  # Estratificado por modalidade
    )

    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]

    # Padronizar nos dados de treino e aplicar nos dois conjuntos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar Isolation Forest apenas no treino
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )
    clf.fit(X_train_scaled)

    # Gerar scores em treino E teste (para posterior comparação com supervisionado)
    # decision_function: maior = mais normal. Invertemos para maior = mais anômalo
    scores_train = -clf.decision_function(X_train_scaled)
    scores_test = -clf.decision_function(X_test_scaled)

    # Previsões binárias (usando threshold treino)
    preds_train = clf.predict(X_train_scaled) == -1
    preds_test = clf.predict(X_test_scaled) == -1

    # Montar dataset com scores e labels
    df_result = df.copy()
    df_result["anomalia_score"] = 0.0
    df_result["anomalia_label"] = False
    df_result["fold"] = "train"

    df_result.loc[idx_train, "anomalia_score"] = scores_train
    df_result.loc[idx_train, "anomalia_label"] = preds_train.astype(bool)
    df_result.loc[idx_train, "fold"] = "train"

    df_result.loc[idx_test, "anomalia_score"] = scores_test
    df_result.loc[idx_test, "anomalia_label"] = preds_test.astype(bool)
    df_result.loc[idx_test, "fold"] = "test"

    # Salvar
    out_path = INTERIM_DIR / "dataset_analitico_com_scores.parquet"
    df_result.to_parquet(out_path, index=False)

    # Estatísticas
    n_anom_train = preds_train.sum()
    n_anom_test = preds_test.sum()
    print(f"\n📊 Isolation Forest com split treino/teste:")
    print(f"  Treino: {len(idx_train):,} linhas, {n_anom_train:,} anomalias")
    print(f"  Teste:  {len(idx_test):,} linhas, {n_anom_test:,} anomalias")
    print(f"  Salvo: {out_path}")
    
    return out_path


def main():
    run_isolation_forest_with_split()


if __name__ == "__main__":
    main()
