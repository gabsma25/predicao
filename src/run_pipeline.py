"""Orquestrador simples para executar pipeline end-to-end.

Etapas:
 1. Consolida CSVs brutos em `data/interim/` (parquets)
 2. Trata os parquets (tipos, normalização, validação de chave)
 3. Gera dataset analítico (`build_features`)
 4. Roda Isolation Forest para gerar `anomalia_score`/`anomalia_label`
 5. Treina classificadores supervisionados usando LABEL HONESTA
    (regras de negócio independentes do Isolation Forest)
 6. Salva modelo e métricas em `models/`

Uso: `python -m src.run_pipeline` ou `from src.run_pipeline import run_all`
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.consolidacao import consolidar_tudo
from src.tratamento import tratar_tudo
from src.features import build_features
from src.anomaly import run_isolation_forest
from src.config import (
    INTERIM_DIR,
    MODELS_DIR,
    SUPERVISIONADO_RANDOM_STATE,
    SUPERVISIONADO_TEST_SIZE,
)


def _build_label_honesta(df: pd.DataFrame) -> pd.Series:
    """
    Cria a label honesta combinando criterio_3 e criterio_4.
    
    Critérios usados:
    - criterio_3: fracionamento (valor 17.500-17.700, limite legal de dispensa)
    - criterio_4: alto valor sem competição (Dispensa/Inexigibilidade > R$ 10M)
    
    Critérios EXCLUÍDOS:
    - criterio_1 (acima P99 modalidade): tautológico (define outlier estatístico)
    - criterio_2 (sem competição): inflaria a label para >60% positivos
    """
    return (
        df["criterio_3_limite_dispensa"].fillna(False).astype(bool) |
        df["criterio_4_alto_valor_sem_comp"].fillna(False).astype(bool)
    ).astype(int)


def _train_supervised(interim_path: Path, models_dir: Path):
    """
    Treina classificador supervisionado com label honesta e features independentes.
    
    Decisões metodológicas:
    - Label = criterio_3 OR criterio_4 (regras de negócio, não saída do IF)
    - Features EXCLUEM as variáveis usadas nos critérios da label
    - Split estratificado por label (preserva proporção de positivos)
    """
    df = pd.read_parquet(interim_path / "dataset_analitico_com_scores.parquet")

    print(f"\n{'='*60}")
    print("TREINO SUPERVISIONADO — LABEL HONESTA + FEATURES INDEPENDENTES")
    print(f"{'='*60}")
    print(f"Dataset: {len(df):,} linhas")

    # === Construção da label honesta ===
    if "criterio_3_limite_dispensa" not in df.columns or "criterio_4_alto_valor_sem_comp" not in df.columns:
        raise RuntimeError(
            "Critérios não encontrados no dataset. "
            "Execute build_features() primeiro."
        )

    y = _build_label_honesta(df)
    print(f"\nLabel honesta (criterio_3 OR criterio_4):")
    print(f"  Positivos: {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"  Negativos: {(1-y).sum():,}")

    # === Features sem leakage ===
    # Variáveis EXCLUÍDAS (entram na composição da label ou são saída do IF):
    # - valor_licitacao, log_valor_licitacao  →  usados em criterio_3 e criterio_4
    # - modalidade_compra                     →  usado em criterio_4
    # - criterio_*                            →  definem a label
    # - candidato_anomalia                    →  agregação dos critérios
    # - anomalia_score, anomalia_label        →  saídas do IF
    
    numeric_features = [
        "n_participantes",
        "n_itens",
        "valor_total_itens",
        "hhi",
        "top1_share",
        "dia_semana",
    ]

    categorical_features = [
        "uf",
        "situacao_licitacao",
        "ano",  # tratado como categoria (2022 / 2023)
    ]

    feature_columns = numeric_features + categorical_features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        # ano pode não existir; remove e segue
        if "ano" in missing_features:
            categorical_features.remove("ano")
            feature_columns = numeric_features + categorical_features
            missing_features.remove("ano")
        if missing_features:
            raise ValueError(f"Features ausentes: {missing_features}")

    # Garantia de não-leakage explícito
    forbidden = {
        "valor_licitacao", "log_valor_licitacao", "modalidade_compra",
        "anomalia_score", "anomalia_label",
        "criterio_1_acima_p99_modalidade", "criterio_2_sem_competicao",
        "criterio_3_limite_dispensa", "criterio_4_alto_valor_sem_comp",
        "candidato_anomalia", "candidato_anomalia_heuristica",
    }
    leakage = set(feature_columns) & forbidden
    if leakage:
        raise ValueError(f"Features com leakage detectadas: {leakage}")

    print(f"\nFeatures usadas ({len(feature_columns)}):")
    print(f"  Numéricas: {numeric_features}")
    print(f"  Categóricas: {categorical_features}")

    X = df[feature_columns].copy()

    # === Split estratificado ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=SUPERVISIONADO_TEST_SIZE,
        random_state=SUPERVISIONADO_RANDOM_STATE,
        stratify=y,
    )
    print(f"\nSplit estratificado:")
    print(f"  Treino: {len(X_train):,} ({y_train.sum():,} positivos = {y_train.mean()*100:.2f}%)")
    print(f"  Teste:  {len(X_test):,} ({y_test.sum():,} positivos = {y_test.mean()*100:.2f}%)")

    # === Pipeline de preprocessamento + modelo ===
    preproc = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = Pipeline([
        ("pre", preproc),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=SUPERVISIONADO_RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("\nTreinando Random Forest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Métricas ===
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "label_method": "criterio_3 OR criterio_4 (regras de negócio)",
        "features_used": feature_columns,
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
    }

    print(f"\n{'='*60}")
    print("MÉTRICAS NO CONJUNTO DE TESTE")
    print(f"{'='*60}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"\nMatriz de confusão:")
    cm = metrics["confusion_matrix"]
    print(f"               Pred Normal    Pred Suspeita")
    print(f"  Real Normal   {cm[0][0]:>10,}     {cm[0][1]:>10,}")
    print(f"  Real Suspeita {cm[1][0]:>10,}     {cm[1][1]:>10,}")

    # === Persistência ===
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "modelo_supervisionado_anomalia.joblib"
    metrics_path = models_dir / "metricas_supervisionado_anomalia.json"

    joblib.dump(clf, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Modelo salvo em: {model_path}")
    print(f"💾 Métricas salvas em: {metrics_path}")
    return model_path, metrics_path


def run_all():
    """Pipeline end-to-end."""
    # 1) Consolidação (CSVs → parquets brutos)
    consolidar_tudo()

    # 2) Tratamento (tipos, normalização, validação)
    tratar_tudo()

    # 3) Features (agregações + feature engineering)
    build_features()

    # 4) Anomalia (Isolation Forest)
    run_isolation_forest()

    # 5) Treino supervisionado (label honesta + features sem leakage)
    model_path, metrics_path = _train_supervised(INTERIM_DIR, MODELS_DIR)

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
    }


if __name__ == "__main__":
    run_all()
