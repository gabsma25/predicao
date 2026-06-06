import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import INTERIM_DIR, CHAVE_LICITACAO


def run_isolation_forest(contamination: float = 0.01, random_state: int = 42):
    path = INTERIM_DIR / "dataset_analitico.parquet"
    df = pd.read_parquet(path)

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

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    clf.fit(Xs)

    # In sklearn, decision_function: larger -> more normal. We invert so larger -> more anomalous
    scores = -clf.decision_function(Xs)
    preds = clf.predict(Xs)  # -1 -> outlier, 1 -> inlier

    df["anomalia_score"] = scores
    df["anomalia_label"] = preds == -1

    out_path = INTERIM_DIR / "dataset_analitico_com_scores.parquet"
    df.to_parquet(out_path, index=False)

    n_anom = int(df["anomalia_label"].sum())
    print(f"Salvo: {out_path} — {n_anom:,} anomalias ({contamination*100:.2f}% contamination)")
    return out_path


def main():
    run_isolation_forest()


if __name__ == "__main__":
    main()
