from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, INTERIM_DIR


def _ensure_reports_dir():
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def gerar_shap_top_anomalias(n=20):
    """Gera análise SHAP para as n licitações com maior probabilidade.

    Salva:
      - reports/shap_top20.json
      - reports/shap_summary_top20.png
      - reports/shap_waterfall_caso1.png

    E imprime uma tabela resumida no console.
    """
    try:
        import joblib
        import shap
    except Exception as e:
        raise RuntimeError("Dependência faltando: instale 'shap' e 'joblib' antes de rodar") from e

    reports_dir = _ensure_reports_dir()

    modelo_path = Path(MODELS_DIR) / "modelo_supervisionado_anomalia.joblib"
    if not modelo_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {modelo_path}")

    modelo = joblib.load(modelo_path)

    # Carrega dataset analítico
    ds_path = Path(INTERIM_DIR) / "dataset_analitico_com_scores.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {ds_path}")

    df = pd.read_parquet(ds_path)

    # Colunas de identificação / exibição esperadas
    id_col = "numero_licitacao"
    modal_col = "modalidade_compra"
    valor_col = "valor_licitacao"

    for c in (id_col, modal_col, valor_col):
        if c not in df.columns:
            raise KeyError(f"Coluna esperada não encontrada no dataset: {c}")

    # Construir X com as colunas que o preprocess espera.
    # Tentativa pragmática: usar as colunas de entrada que o pipeline recebeu,
    # isto é, usar modelo[:-1].feature_names_in_ quando disponível.
    pre = modelo[:-1]
    try:
        feature_inputs = pre.feature_names_in_
    except Exception:
        # fallback: usar todas colunas exceto id/modal/valor
        feature_inputs = [c for c in df.columns if c not in {id_col, modal_col, valor_col}]

    X = df.loc[:, feature_inputs].copy()

    # Predições de probabilidade no dataset completo
    proba = modelo.predict_proba(X)[:, 1]

    # Seleciona top-n
    top_idx = np.argsort(proba)[-n:][::-1]

    # Transforma X apenas com o preprocessador (modelo[:-1])
    X_transformed = pre.transform(X)
    # tornar denso se for sparse
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # nomes das features após transformação
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        # gera nomes genéricos
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    X_top = X.iloc[top_idx]
    X_top_trans = X_transformed[top_idx]
    proba_top = proba[top_idx]

    # Explainer no classificador (etapa 'clf')
    clf = modelo.named_steps.get('clf', None)
    if clf is None:
        # tentar último passo do pipeline
        clf = modelo.steps[-1][1]

    explainer = shap.TreeExplainer(clf)

    # calcula shap apenas para as top-n amostras
    shap_values_raw = explainer.shap_values(X_top_trans)

    # Normalizar saídas para forma (n, n_features)
    if isinstance(shap_values_raw, (list, tuple)) and len(shap_values_raw) == 2:
        # lista por classe -> pegar a classe positiva
        shap_vals = np.array(shap_values_raw[1])
    elif hasattr(shap_values_raw, 'ndim') and shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
        shap_vals = shap_values_raw[:, :, 1]
    elif hasattr(shap_values_raw, 'ndim') and shap_values_raw.ndim == 2:
        shap_vals = shap_values_raw
    else:
        # fallback: tentar converter
        shap_vals = np.asarray(shap_values_raw)
        if shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
            shap_vals = shap_vals[:, :, 1]

    # Guarda JSON com top-5 contribuições por instância
    out = []
    for i, idx in enumerate(top_idx):
        row_id = int(df.iloc[idx][id_col]) if pd.notna(df.iloc[idx][id_col]) else str(df.iloc[idx][id_col])
        row_modal = df.iloc[idx][modal_col]
        row_valor = float(df.iloc[idx][valor_col]) if pd.notna(df.iloc[idx][valor_col]) else None
        vals = shap_vals[i]
        abs_order = np.argsort(np.abs(vals))[::-1]
        top5 = []
        for j in abs_order[:5]:
            top5.append([str(feature_names[j]), float(vals[j])])
        out.append({
            "numero_licitacao": row_id,
            "modalidade_compra": str(row_modal),
            "valor_licitacao": row_valor,
            "probabilidade": float(proba_top[i]),
            "top_5_contribuicoes": top5,
        })

    with open(reports_dir / "shap_top20.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Plots
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_top_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary_top20.png")
    plt.close()

    # Waterfall para o caso 1 (maior probabilidade)
    # construindo Explanation para o primeiro caso
    first_vals = shap_vals[0]
    try:
        base_value = explainer.expected_value[1]
    except Exception:
        # fallback
        base_value = float(np.mean(modelo.predict_proba(X)[:, 1]))

    expl = shap.Explanation(values=first_vals, base_values=base_value, data=X_top_trans[0], feature_names=feature_names)
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_waterfall_caso1.png")
    plt.close()

    # Imprime tabela resumida
    header = "numero_licitacao | modalidade | valor | proba | top_feature_1 | top_feature_2 | top_feature_3"
    print(header)
    for i in range(len(out)):
        row = out[i]
        top_feats = [t[0] for t in row["top_5_contribuicoes"][:3]]
        print(f"{row['numero_licitacao']} | {row['modalidade_compra']} | {row['valor_licitacao']:.2f} | {row['probabilidade']:.4f} | {top_feats[0]} | {top_feats[1]} | {top_feats[2]}")

    # Confirmar existência dos arquivos
    files = [reports_dir / "shap_top20.json", reports_dir / "shap_summary_top20.png", reports_dir / "shap_waterfall_caso1.png"]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo esperado não gerado: {p}")

    print('\nArquivos gerados:')
    for p in files:
        print(str(p))

    return out
