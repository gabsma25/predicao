import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import ANOMALIA_CONTAMINATION, ANOMALIA_RANDOM_STATE, INTERIM_DIR, SUPERVISIONADO_TEST_SIZE


def _load_anomaly_frame():
    path = INTERIM_DIR / "dataset_analitico.parquet"
    return pd.read_parquet(path)


def _prepare_features(df: pd.DataFrame):
    # Features to use
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
    return df, X, feats


def run_isolation_forest_with_split(
    test_size: float = SUPERVISIONADO_TEST_SIZE,
    contamination: float = ANOMALIA_CONTAMINATION,
    random_state: int = ANOMALIA_RANDOM_STATE,
):
    df = _load_anomaly_frame()
    df, X, _ = _prepare_features(df)

    idx_train, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(X_train_scaled)

    df.loc[idx_train, "anomalia_score"] = -clf.decision_function(X_train_scaled)
    df.loc[idx_test, "anomalia_score"] = -clf.decision_function(X_test_scaled)
    df.loc[idx_train, "anomalia_label"] = clf.predict(X_train_scaled) == -1
    df.loc[idx_test, "anomalia_label"] = clf.predict(X_test_scaled) == -1
    df["split_dataset"] = "train"
    df.loc[idx_test, "split_dataset"] = "test"

    out_path = INTERIM_DIR / "dataset_analitico_com_scores.parquet"
    df.to_parquet(out_path, index=False)

    n_anom = int(df["anomalia_label"].sum())
    print(
        f"Salvo: {out_path} — {n_anom:,} anomalias "
        f"(treino={len(idx_train):,}, teste={len(idx_test):,}, contamination={contamination*100:.2f}%)"
    )
    return out_path


def run_isolation_forest(contamination: float = ANOMALIA_CONTAMINATION, random_state: int = ANOMALIA_RANDOM_STATE):
    # Mantém compatibilidade com o notebook 03, mas agora usa split interno.
    return run_isolation_forest_with_split(
        test_size=SUPERVISIONADO_TEST_SIZE,
        contamination=contamination,
        random_state=random_state,
    )


def main():
    run_isolation_forest()


if __name__ == "__main__":
    main()
