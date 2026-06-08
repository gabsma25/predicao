"""
Análise de ablação: testar impacto de remover features suspeitas
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, auc, precision_recall_curve

from src.config import INTERIM_DIR


def run_ablation_experiment(
    feature_set_name: str,
    features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Treina os três modelos com um conjunto específico de features.
    
    Retorna dicionário com métricas (F1, ROC-AUC, PR-AUC) para cada modelo.
    """
    
    # Prepara dados com as features selecionadas
    X_train_subset = X_train[features].fillna(0)
    X_test_subset = X_test[features].fillna(0)
    
    # Padroniza
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_test_scaled = scaler.transform(X_test_subset)
    
    results = {"feature_set": feature_set_name, "num_features": len(features)}
    
    # --- KNN com tuning ---
    best_k = 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for k in [3, 5, 7, 10, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_validate(
            knn, X_train_scaled, y_train,
            cv=skf,
            scoring=['f1'],
            n_jobs=-1
        )
        if scores['test_f1'].mean() > results.get('knn_cv_f1', -1):
            best_k = k
            results['knn_cv_f1'] = scores['test_f1'].mean()
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
    
    results['knn_f1'] = f1_score(y_test, y_pred_knn)
    results['knn_roc_auc'] = roc_auc_score(y_test, y_proba_knn)
    results['knn_pr_auc'] = auc(*precision_recall_curve(y_test, y_proba_knn)[:2])
    results['knn_precision'] = precision_score(y_test, y_pred_knn)
    results['knn_recall'] = recall_score(y_test, y_pred_knn)
    results['knn_best_k'] = best_k
    
    # --- Decision Tree com tuning ---
    best_depth = 5
    for depth in [3, 5, 7, 10, 15]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=random_state, min_samples_split=10)
        scores = cross_validate(
            dt, X_train_scaled, y_train,
            cv=skf,
            scoring=['f1'],
            n_jobs=-1
        )
        if scores['test_f1'].mean() > results.get('dt_cv_f1', -1):
            best_depth = depth
            results['dt_cv_f1'] = scores['test_f1'].mean()
    
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=random_state, min_samples_split=10)
    dt.fit(X_train_scaled, y_train)
    y_pred_dt = dt.predict(X_test_scaled)
    y_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]
    
    results['dt_f1'] = f1_score(y_test, y_pred_dt)
    results['dt_roc_auc'] = roc_auc_score(y_test, y_proba_dt)
    results['dt_pr_auc'] = auc(*precision_recall_curve(y_test, y_proba_dt)[:2])
    results['dt_precision'] = precision_score(y_test, y_pred_dt)
    results['dt_recall'] = recall_score(y_test, y_pred_dt)
    results['dt_best_depth'] = best_depth
    
    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    results['rf_f1'] = f1_score(y_test, y_pred_rf)
    results['rf_roc_auc'] = roc_auc_score(y_test, y_proba_rf)
    results['rf_pr_auc'] = auc(*precision_recall_curve(y_test, y_proba_rf)[:2])
    results['rf_precision'] = precision_score(y_test, y_pred_rf)
    results['rf_recall'] = recall_score(y_test, y_pred_rf)
    
    return results


def main():
    """Executa análise de ablação completa"""
    
    # Carrega dados
    path = INTERIM_DIR / "dataset_analitico_com_scores.parquet"
    df = pd.read_parquet(path)
    
    df_train = df[df['fold'] == 'train'].copy()
    df_test = df[df['fold'] == 'test'].copy()
    
    # Features base
    base_features = [
        'log_valor_licitacao',
        'n_participantes',
        'n_itens',
        'hhi',
        'top1_share',
        'valor_total_itens',
    ]
    
    # Labels
    y_train = df_train['anomalia_label'].astype(int)
    y_test = df_test['anomalia_label'].astype(int)
    X_train = df_train
    X_test = df_test
    
    # Executa ablação
    ablation_results = []
    
    # Experimento 1: Todas as features
    print("=" * 80)
    print("EXPERIMENTO 1: Todas as features base")
    print("=" * 80)
    result = run_ablation_experiment(
        feature_set_name="Todas (baseline)",
        features=base_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    ablation_results.append(result)
    print_result(result)
    
    # Experimento 2-N: Remover cada feature uma de cada vez
    for feature_to_remove in base_features:
        remaining_features = [f for f in base_features if f != feature_to_remove]
        print("\n" + "=" * 80)
        print(f"EXPERIMENTO: Removendo '{feature_to_remove}'")
        print(f"Features: {remaining_features}")
        print("=" * 80)
        result = run_ablation_experiment(
            feature_set_name=f"Sem {feature_to_remove}",
            features=remaining_features,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        ablation_results.append(result)
        print_result(result)
    
    # Salva tabela comparativa
    results_df = pd.DataFrame(ablation_results)
    out_path = INTERIM_DIR / "ablacao_resultados.parquet"
    results_df.to_parquet(out_path, index=False)
    print("\n" + "=" * 80)
    print(f"Resultados salvos em: {out_path}")
    print("=" * 80)
    
    # Exibe tabela resumida
    print("\nRESUMO COMPARATIVO - Métrica F1 por modelo:")
    print("-" * 80)
    summary_cols = ['feature_set', 'num_features', 'knn_f1', 'dt_f1', 'rf_f1']
    print(results_df[summary_cols].to_string(index=False))
    
    return results_df


def print_result(result: Dict[str, float]):
    """Exibe resultado de um experimento de forma legível"""
    print(f"\n  KNN (k={result.get('knn_best_k', '?')}):")
    print(f"    F1:       {result.get('knn_f1', 0):.4f}")
    print(f"    ROC-AUC:  {result.get('knn_roc_auc', 0):.4f}")
    print(f"    PR-AUC:   {result.get('knn_pr_auc', 0):.4f}")
    print(f"    Precision: {result.get('knn_precision', 0):.4f}")
    print(f"    Recall:    {result.get('knn_recall', 0):.4f}")
    
    print(f"\n  Decision Tree (depth={result.get('dt_best_depth', '?')}):")
    print(f"    F1:       {result.get('dt_f1', 0):.4f}")
    print(f"    ROC-AUC:  {result.get('dt_roc_auc', 0):.4f}")
    print(f"    PR-AUC:   {result.get('dt_pr_auc', 0):.4f}")
    print(f"    Precision: {result.get('dt_precision', 0):.4f}")
    print(f"    Recall:    {result.get('dt_recall', 0):.4f}")
    
    print(f"\n  Random Forest:")
    print(f"    F1:       {result.get('rf_f1', 0):.4f}")
    print(f"    ROC-AUC:  {result.get('rf_roc_auc', 0):.4f}")
    print(f"    PR-AUC:   {result.get('rf_pr_auc', 0):.4f}")
    print(f"    Precision: {result.get('rf_precision', 0):.4f}")
    print(f"    Recall:    {result.get('rf_recall', 0):.4f}")


if __name__ == "__main__":
    main()
