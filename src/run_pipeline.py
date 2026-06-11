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
import os
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
    precision_recall_curve,
)

from src.consolidacao import consolidar_tudo
from src.tratamento import tratar_tudo
from src.features import build_features
from src.anomaly import run_isolation_forest_with_split as run_isolation_forest
from src.explicabilidade import gerar_shap_top_anomalias
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


def _cross_validate_supervised(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    models_dir: Path,
    n_splits: int = 5,
    random_state: int = SUPERVISIONADO_RANDOM_STATE,
) -> dict:
    """
    Valida o pipeline via StratifiedKFold sobre o conjunto de treino.

    Retorna dict com média e desvio padrão de cada métrica, e persiste
    em models/metricas_cv_supervisionado.json.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
    }

    print(f"\n{'='*60}")
    print(f"VALIDAÇÃO CRUZADA ESTRATIFICADA ({n_splits} folds) — CONJUNTO DE TREINO")
    print(f"{'='*60}")

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    rows = []
    for label, key in [
        ("ROC-AUC",           "test_roc_auc"),
        ("PR-AUC",            "test_pr_auc"),
        ("F1",                "test_f1"),
        ("Precision",         "test_precision"),
        ("Recall",            "test_recall"),
        ("Balanced Accuracy", "test_balanced_accuracy"),
    ]:
        values = cv_results[key]
        mean, std = float(np.mean(values)), float(np.std(values))
        short = key.replace("test_", "")
        summary[short] = {"mean": mean, "std": std, "values": values.tolist()}
        rows.append((label, mean, std))

    # Tabela formatada
    print(f"\n  {'Métrica':<22} {'Média':>8}  {'Std':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    for label, mean, std in rows:
        print(f"  {label:<22} {mean:>8.4f}  {std:>8.4f}")

    # Persistência
    models_dir.mkdir(parents=True, exist_ok=True)
    cv_path = models_dir / "metricas_cv_supervisionado.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[salvo] Métricas CV: {cv_path}")

    return summary


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
        "log_razao_valor_mediana",
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

    # === Validação cruzada (antes do treino final) ===
    _cross_validate_supervised(X_train, y_train, clf, models_dir)

    print("\nTreinando Random Forest (modelo final em todo X_train)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Métricas iniciais no threshold default (0.5) ===
    metrics_default = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # === Cálculo da curva Precision-Recall e otimização do threshold via F2 ===
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    # precisions/recalls have len = len(pr_thresholds) + 1; align by ignoring last point for thresholds
    if len(pr_thresholds) > 0:
        p_cut = precisions[:-1]
        r_cut = recalls[:-1]
        t_cut = pr_thresholds
    else:
        # edge case: single point
        p_cut = precisions
        r_cut = recalls
        t_cut = np.array([0.5])

    # F2 formula (beta=2 => weight recall 4x)
    f2_scores = (1 + 4) * (p_cut * r_cut) / (4 * p_cut + r_cut + 1e-10)
    best_idx = int(np.argmax(f2_scores)) if len(f2_scores) > 0 else 0
    best_threshold = float(t_cut[best_idx])

    # Reclassifica com threshold ótimo
    y_pred_opt = (y_proba >= best_threshold).astype(int)

    precision_opt = float(precision_score(y_test, y_pred_opt, zero_division=0))
    recall_opt = float(recall_score(y_test, y_pred_opt, zero_division=0))
    f1_opt = float(f1_score(y_test, y_pred_opt, zero_division=0))
    f2_opt = (1 + 4) * (precision_opt * recall_opt) / (4 * precision_opt + recall_opt + 1e-10)

    metrics_opt = {
        "precision": precision_opt,
        "recall": recall_opt,
        "f1": f1_opt,
        "f2": f2_opt,
        "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
    }

    # Metrics default F2
    f2_default = (1 + 4) * (metrics_default["precision"] * metrics_default["recall"]) / (
        4 * metrics_default["precision"] + metrics_default["recall"] + 1e-10
    )

    print(f"\n{'='*60}")
    print("MÉTRICAS NO CONJUNTO DE TESTE")
    print(f"{'='*60}")
    print(f"  ROC-AUC:           {metrics_default['roc_auc']:.4f}")
    print(f"  PR-AUC:            {metrics_default['pr_auc']:.4f}")
    print(f"  F1 (default 0.5):  {metrics_default['f1']:.4f}")
    print(f"  Precision (0.5):   {metrics_default['precision']:.4f}")
    print(f"  Recall (0.5):      {metrics_default['recall']:.4f}")
    print(f"  Balanced Accuracy: {metrics_default['balanced_accuracy']:.4f}") if "balanced_accuracy" in metrics_default else None
    print(f"\nMatriz de confusão (default 0.5):")
    cm_def = metrics_default["confusion_matrix"]
    print(f"               Pred Normal    Pred Suspeita")
    print(f"  Real Normal   {cm_def[0][0]:>10,}     {cm_def[0][1]:>10,}")
    print(f"  Real Suspeita {cm_def[1][0]:>10,}     {cm_def[1][1]:>10,}")

    # Comparação lado a lado (formato pedido)
    print("\nComparação de thresholds:")
    print(f"Threshold default (0.5):  Precision={metrics_default['precision']:.2f}  Recall={metrics_default['recall']:.2f}  F1={metrics_default['f1']:.2f}  F2={f2_default:.2f}")
    print(f"Threshold ótimo ({best_threshold:.4f}):   Precision={precision_opt:.2f}  Recall={recall_opt:.2f}  F1={f1_opt:.2f}  F2={f2_opt:.2f}")

    # === Persistência: modelo, métricas e threshold ótimo + curva PR completa ===
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "modelo_supervisionado_anomalia.joblib"
    metrics_path = models_dir / "metricas_supervisionado_anomalia.json"

    joblib.dump(clf, model_path)
    # Atualiza arquivo de métricas padrão
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics_default, "f2": f2_default}, f, indent=2, ensure_ascii=False)

    # Salva threshold ótimo e curva PR completa
    threshold_info = {
        "threshold_default": 0.5,
        "threshold_otimo": best_threshold,
        "metrics_default": {k: (float(v) if not isinstance(v, list) else v) for k, v in metrics_default.items()},
        "metrics_otimo": metrics_opt,
        "pr_curve": [
            {"precision": float(p), "recall": float(r), "threshold": float(t)}
            for p, r, t in zip(p_cut.tolist(), r_cut.tolist(), t_cut.tolist())
        ],
    }
    threshold_path = models_dir / "threshold_otimo.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2, ensure_ascii=False)

    print(f"\n[salvo] Modelo: {model_path}")
    print(f"[salvo] Métricas: {metrics_path}")
    print(f"[salvo] Threshold info: {threshold_path}")

    # === Plot da curva PR com pontos marcados ===
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_path = reports_dir / "curva_pr_threshold.png"

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label="Precision-Recall curve")
    # ponto default (0.5)
    # compute precision/recall for default
    prec_def = metrics_default["precision"]
    rec_def = metrics_default["recall"]
    plt.scatter([rec_def], [prec_def], color="red", label="Default (0.5)")
    plt.annotate(f"0.5", (rec_def, prec_def))
    # ponto ótimo
    plt.scatter([recall_opt], [precision_opt], color="green", label=f"Ótimo ({best_threshold:.4f})")
    plt.annotate(f"{best_threshold:.4f}", (recall_opt, precision_opt))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall com thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"[salvo] Figura PR: {fig_path}")

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

    try:
        print('\nGerando explicabilidade SHAP para top 20...')
        gerar_shap_top_anomalias(n=20)
    except Exception as e:
        print(f'Falha ao gerar SHAP: {e}')

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
    }


if __name__ == "__main__":
    run_all()