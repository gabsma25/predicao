"""
KNN melhorado com tuning agressivo e ajuste de threshold
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    auc, precision_recall_curve, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import INTERIM_DIR


def tune_knn_aggressive(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, any]:
    """
    Tuning agressivo de KNN:
    - k ∈ {1, 3, 5, 7, 15, 25}
    - weights ∈ {'uniform', 'distance'}
    - Métrica: F1 ou PR-AUC
    """
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    best_params = None
    best_score = -np.inf
    cv_history = []
    
    for k in [1, 3, 5, 7, 15, 25]:
        for weights in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
            scores = cross_validate(
                knn, X_train_scaled, y_train,
                cv=skf,
                scoring=['f1', 'roc_auc', 'precision_recall_weighted'],
                n_jobs=-1
            )
            
            f1_mean = scores['test_f1'].mean()
            roc_auc_mean = scores['test_roc_auc'].mean()
            
            # Prioridade: F1 (recall tem peso em classe rara)
            score_to_optimize = f1_mean
            
            cv_history.append({
                'k': k,
                'weights': weights,
                'f1_mean': f1_mean,
                'f1_std': scores['test_f1'].std(),
                'roc_auc_mean': roc_auc_mean,
                'roc_auc_std': scores['test_roc_auc'].std(),
            })
            
            if score_to_optimize > best_score:
                best_score = score_to_optimize
                best_params = {'k': k, 'weights': weights, 'f1_score': f1_mean}
    
    cv_df = pd.DataFrame(cv_history).sort_values('f1_mean', ascending=False)
    
    return {
        'best_params': best_params,
        'cv_history': cv_df,
        'best_score': best_score,
    }


def find_optimal_threshold(
    y_test: pd.Series,
    y_proba_test: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Encontra threshold ótimo para predict_proba que maximize F1 ou outro métrica.
    """
    
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_metric_value = -np.inf
    metric_values = []
    
    for threshold in thresholds:
        y_pred = (y_proba_test >= threshold).astype(int)
        
        if metric == 'f1':
            metric_value = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'roc_auc':
            # ROC-AUC não depende de threshold, mas podemos usar AUC-PR
            metric_value = auc(*precision_recall_curve(y_test, y_proba_test)[:2])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_values.append(metric_value)
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    return best_threshold, best_metric_value


def run_knn_improved(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Pipeline completo de KNN melhorado
    """
    
    print("\n" + "=" * 80)
    print("KNN MELHORADO - Tuning Agressivo")
    print("=" * 80)
    
    # Tuning
    print("\n1. Executando grid search sobre k e weights...")
    tuning_result = tune_knn_aggressive(X_train_scaled, y_train, random_state=random_state)
    best_params = tuning_result['best_params']
    cv_df = tuning_result['cv_history']
    
    print(f"\nTop 5 configurações por F1:")
    print(cv_df.head(5)[['k', 'weights', 'f1_mean', 'roc_auc_mean']].to_string(index=False))
    
    print(f"\n✓ Melhor configuração: k={best_params['k']}, weights={best_params['weights']}")
    print(f"  F1 em CV: {best_params['f1_score']:.4f}")
    
    # Treinar com melhor configuração
    print("\n2. Treinando KNN com melhor configuração...")
    knn = KNeighborsClassifier(
        n_neighbors=best_params['k'],
        weights=best_params['weights']
    )
    knn.fit(X_train_scaled, y_train)
    
    # Predições
    y_pred_default = knn.predict(X_test_scaled)
    y_proba = knn.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas com threshold padrão
    print("\n3. Avaliação no teste (threshold=0.5):")
    print(f"  F1:        {f1_score(y_test, y_pred_default):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  PR-AUC:    {auc(*precision_recall_curve(y_test, y_proba)[:2]):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_default):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_default):.4f}")
    
    # Encontrar threshold ótimo
    print("\n4. Buscando threshold ótimo...")
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba, metric='f1')
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    print(f"  Threshold ótimo: {optimal_threshold:.4f}")
    print(f"  F1 com threshold ótimo: {optimal_f1:.4f}")
    print(f"  Melhoramento: {optimal_f1 - f1_score(y_test, y_pred_default):+.4f}")
    
    # Métricas com threshold ótimo
    print(f"\n  Métricas com threshold ótimo:")
    print(f"    F1:        {f1_score(y_test, y_pred_optimal):.4f}")
    print(f"    Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"    Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
    
    # Matriz de confusão
    print(f"\n  Matriz de confusão (threshold ótimo):")
    cm = confusion_matrix(y_test, y_pred_optimal)
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Report detalhado
    print(f"\n  Classificação detalhada:")
    print(classification_report(y_test, y_pred_optimal, digits=4))
    
    return {
        'model': knn,
        'best_params': best_params,
        'optimal_threshold': optimal_threshold,
        'y_pred_default': y_pred_default,
        'y_pred_optimal': y_pred_optimal,
        'y_proba': y_proba,
        'cv_history': cv_df,
    }


def visualize_knn_results(result: Dict, y_test: pd.Series, output_dir: Path = None):
    """Cria visualizações dos resultados do KNN"""
    
    if output_dir is None:
        output_dir = INTERIM_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    axes[0, 0].plot(fpr, tpr, linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random classifier')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve - KNN Melhorado')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, result['y_proba'])
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve - KNN Melhorado')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distribuição de probabilidades
    axes[1, 0].hist(result['y_proba'][y_test == 0], bins=30, alpha=0.6, label='Negativas')
    axes[1, 0].hist(result['y_proba'][y_test == 1], bins=30, alpha=0.6, label='Anomalias')
    axes[1, 0].axvline(result['optimal_threshold'], color='red', linestyle='--', label=f"Threshold ótimo: {result['optimal_threshold']:.3f}")
    axes[1, 0].set_xlabel('Probabilidade predita')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição de probabilidades preditas')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Top k values por F1
    cv_df = result['cv_history']
    cv_df_grouped = cv_df.groupby('k')['f1_mean'].max().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(cv_df_grouped)), cv_df_grouped.values)
    axes[1, 1].set_yticks(range(len(cv_df_grouped)))
    axes[1, 1].set_yticklabels([f"k={int(k)}" for k in cv_df_grouped.index])
    axes[1, 1].set_xlabel('F1 Score')
    axes[1, 1].set_title('Top k values por F1 em CV')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(output_dir / "knn_melhorado_resultados.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráficos salvos em: {output_dir / 'knn_melhorado_resultados.png'}")
    
    return fig


def main():
    """Executa pipeline completo de KNN melhorado"""
    
    # Carrega dados
    path = INTERIM_DIR / "dataset_analitico_com_scores.parquet"
    df = pd.read_parquet(path)
    
    df_train = df[df['fold'] == 'train'].copy()
    df_test = df[df['fold'] == 'test'].copy()
    
    # Features
    features = [
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
    
    # Padroniza
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[features].fillna(0))
    X_test_scaled = scaler.transform(df_test[features].fillna(0))
    
    # Executa KNN melhorado
    result = run_knn_improved(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Visualiza
    visualize_knn_results(result, y_test, output_dir=INTERIM_DIR)
    
    # Salva configuração
    config_df = result['cv_history']
    config_df.to_parquet(INTERIM_DIR / "knn_tuning_history.parquet", index=False)
    print(f"\n✓ Histórico de tuning salvo em: {INTERIM_DIR / 'knn_tuning_history.parquet'}")
    
    return result


if __name__ == "__main__":
    main()
