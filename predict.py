import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── File path defaults (edit here if files live elsewhere) ───────────────────

MODELS_FILE    = "Isolated_Ensemble_Models.pkl"
SELECTORS_FILE = "Isolated_Ensemble_Selectors.pkl"
DISAMBIG_MODEL = "Disambiguation_Model.pkl"
DISAMBIG_SEL   = "Disambiguation_Selector.pkl"
TRAIN_REF_FILE = "training_reference_matrix.pkl"
GENE_LIST_FILE = "isolated_gene_list.txt"
GPL96_ANNOT    = "GPL96_annot.txt"       
GPL13667_ANNOT = "GPL13667_annot.txt"    

# ── Per-dataset configuration ─────────────────────────────────────────────────

DRUG_MAP = {
    0: "Control", 1: "Cisplatin", 2: "Doxorubicin",
    3: "Lapatinib", 4: "Vorinostat", 5: "Sunitinib",
}

DATASET_CONFIGS = {
    "gse3926": {
        "description":        "GSE3926 — GPL96 (HGU133A), doxorubicin study",
        "expr_file":          "new_samples_gse3926.csv",
        "meta_file":          "sample_metadata_gse3926.csv",
        "already_log2":       False,
        "detection_threshold": 0.50,
        "valid_conditions":   ["untreated", "treated_dox", "resistant"],
        "target_drug_label":  2,  # Tracking Doxorubicin detection
        "cross_platform":     False,
    },
    "gse69644": {
        "description":        "GSE69644 — GPL13667 (HG-U219), cisplatin study, HK-2 cells",
        "expr_file":          "new_samples_gse69644.csv",
        "meta_file":          "sample_metadata_gse69644.csv",
        "already_log2":       True,
        "detection_threshold": 0.5,
        "valid_conditions":   ["untreated", "treated_cis_24h"],
        "target_drug_label":  1,  # Tracking Cisplatin detection
        "cross_platform":     True,
    },
}


# ── Auto-detect dataset ───────────────────────────────────────────────────────

def detect_dataset(filepath):
    name = os.path.basename(filepath).lower()
    for key in DATASET_CONFIGS:
        if key in name:
            return key
    return None


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def log2_if_raw(df):
    if df.values.max() > 50:
        return np.log2(df.astype(float) + 1)
    return df.astype(float)

def zscore_columns(df):
    return (df - df.mean()) / (df.std() + 1e-8)

def zscore_rows(df):
    return df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

def extract_cell_line(title):
    return title.split("_")[0].strip()

def clean_columns(df):
    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
    return df


# ── Cross-platform helpers ────────────────────────────────────────────────────

def load_probe_to_gene(annot_file, probe_col="ID", gene_col="Gene Symbol"):
    df = pd.read_csv(annot_file, sep="\t", comment="#", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[[probe_col, gene_col]].dropna()
    df = df[df[gene_col].str.strip() != ""]
    mapping = {}
    for _, row in df.iterrows():
        probe = str(row[probe_col]).strip()
        gene  = str(row[gene_col]).split("///")[0].strip().upper()
        if gene:
            mapping[probe] = gene
    return mapping

def collapse_to_gene_symbols(expr_df, probe_to_gene):
    mapped   = [p for p in expr_df.columns if p in probe_to_gene]
    sub         = expr_df[mapped].copy()
    sub.columns = [probe_to_gene[p] for p in mapped]
    return sub.T.groupby(level=0).mean().T

def apply_combat(X_train_ref, new_data):
    try:
        from combat.pycombat import pycombat
    except ImportError:
        print("  ⚠️  pycombat not installed — skipping ComBat (pip install combat)")
        return new_data
    common = list(set(X_train_ref.columns) & set(new_data.columns))
    if not common:
        return new_data
    combined     = pd.concat([X_train_ref[common], new_data[common]], axis=0).T
    batch_vector = [1] * len(X_train_ref) + [2] * len(new_data)
    harmonized   = pycombat(combined, batch_vector).T
    return harmonized.iloc[len(X_train_ref):]


# ── Two-stage prediction ──────────────────────────────────────────────────────

def run_two_stage(delta_df, ensemble_models, ensemble_selectors,
                  disambig_model, disambig_selector, threshold):
    raw_probs = {}
    for label, model in ensemble_models.items():
        X_sel            = ensemble_selectors[label].transform(delta_df)
        raw_probs[label] = model.predict_proba(X_sel)[:, 1]

    results = []
    for i in range(len(delta_df)):
        fired = [lbl for lbl in ensemble_models if raw_probs[lbl][i] > threshold]
        if len(fired) == 0:
            results.append({"pred": 0, "method": "no-signal",
                            "raw_probs": {k: raw_probs[k][i] for k in raw_probs}})
        elif len(fired) == 1:
            results.append({"pred": fired[0], "method": "unambiguous",
                            "raw_probs": {k: raw_probs[k][i] for k in raw_probs}})
        else:
            d_probs      = disambig_model.predict_proba(disambig_selector.transform(delta_df.iloc[[i]]))[0]
            fired_scores = {c: p for c, p in zip(disambig_model.classes_, d_probs) if c in fired}
            winner       = max(fired_scores, key=fired_scores.get)
            results.append({"pred": winner, "method": f"disambiguated (fired: {fired})",
                            "raw_probs": {k: raw_probs[k][i] for k in raw_probs},
                            "disambig_scores": fired_scores})
    return results


DRUG_COLORS = {
    0: "#aaaaaa", 1: "#4e9af1", 2: "#e05c5c",
    3: "#57b857", 4: "#f0a832", 5: "#9b6fd4",
}


# ── Bar-chart: visualization mapping resistance outcomes ──────────────────────

# ── Bar-chart: visualization mapping resistance outcomes ──────────────────────

def save_score_chart(delta_df, results, threshold, label_fn, title, out_path, target_lbl):
    drug_keys   = [k for k in sorted(DRUG_MAP) if k > 0]
    n_drugs     = len(drug_keys)
    n_samples   = len(delta_df)

    # Make the graph slightly wider to fit the text labels on the right
    fig_h = max(4, n_samples * (n_drugs * 0.35 + 0.6) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h)) 

    bar_h   = 0.12
    spacing = bar_h * (n_drugs + 1.5)
    y_ticks = []
    y_labels = []

    for i, sid in enumerate(delta_df.index):
        r        = results[i]
        pred_lbl = r["pred"]
        base_y   = i * spacing

        target_score = r["raw_probs"].get(target_lbl, 0.0)
        status = "[SENSITIVE]" if target_score >= threshold else "[RESISTANT]"

        for j, lbl in enumerate(drug_keys):
            score  = r["raw_probs"].get(lbl, 0.0)
            y_pos  = base_y + j * bar_h
            color  = DRUG_COLORS[lbl]
            edge   = "black" if lbl == pred_lbl else "none"
            lw     = 1.8    if lbl == pred_lbl else 0

            # 1. Draw the actual bar
            ax.barh(y_pos, score * 100, height=bar_h * 0.85,
                    color=color, edgecolor=edge, linewidth=lw, zorder=3)
            
            # 2. NEW: Add the explicit percentage text right beside the bar
            # We only show it if the score is greater than 0% so it stays clean
            if score > 0.001: 
                ax.text(score * 100 + 1.5, y_pos, f"{score*100:.1f}%", 
                        va='center', fontsize=8, color='#333333', zorder=5)

        centre = base_y + (n_drugs - 1) * bar_h / 2
        y_ticks.append(centre)
        y_labels.append(f"{label_fn(sid)}\n{status}")

    ax.axvline(threshold * 100, color="black", linewidth=1.2,
               linestyle="--", zorder=4, label=f"Threshold ({threshold*100:.0f}%)")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("Model Signalling / Confidence Score (%)", fontsize=11)
    ax.set_xlim(0, 115) # Extended the max limit slightly so text doesn't get cut off
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=DRUG_COLORS[k], label=DRUG_MAP[k]) for k in drug_keys]
    threshold_line = plt.Line2D([0], [0], color="black", linewidth=1.2,
                                linestyle="--", label=f"Threshold ({threshold*100:.0f}%)")
    ax.legend(handles=patches + [threshold_line], loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visual Report Saved → {out_path}")


# ── Result printers ───────────────────────────────────────────────────────────

def print_results_multi(delta_df, results, meta, paired_meta, threshold, cfg, dataset_key):
    target_lbl  = cfg["target_drug_label"]
    target_name = DRUG_MAP[target_lbl]

    print("═" * 80)
    print(f"  FINAL PIPELINE ANALYSIS: RESISTANCE PROFILE STATUS FOR {target_name.upper()}")
    print("═" * 80)
    print(f"{'Cell line':<12} {'Sample ID':<12} {'Model Call':<14} {'Target Score':<14} {'Resistance Status'}")
    print("─" * 80)
    
    for i, sid in enumerate(delta_df.index):
        r         = results[i]
        cell_line = paired_meta[sid]["cell_line"]
        score     = r["raw_probs"].get(target_lbl, 0.0)
        
        # Core resistance logic check
        status = "SENSITIVE" if score >= threshold else "RESISTANT"
        
        print(f"{cell_line:<12} {str(sid):<12} {DRUG_MAP[r['pred']]:<14} {score:>11.2%}    {status:<15}")
        
    print("─" * 80)
    out_path = f"resistance_scores_{dataset_key}.png"
    save_score_chart(delta_df, results, threshold,
                     label_fn=lambda sid: f"{paired_meta[sid]['cell_line']} ({sid})",
                     title=f"Drug Functional Resistance Profiling Matrix — {dataset_key.upper()}",
                     out_path=out_path, target_lbl=target_lbl)


def print_results_single(delta_df, results, sample_label, threshold, description, cfg, dataset_key):
    r          = results[0]
    target_lbl = cfg["target_drug_label"]
    score      = r["raw_probs"].get(target_lbl, 0.0)
    status     = "SENSITIVE" if score >= threshold else "RESISTANT"

    print("═" * 70)
    print(f"  RESULT — {description}")
    print("═" * 70)
    print(f"  Target Drug Detected : {DRUG_MAP[target_lbl]}")
    print(f"  Model Confidence     : {score:.2%}")
    print(f"  Functional Status    : {status}")
    print(f"  Background Note      : {r['method']}")
    print()

    out_path = f"resistance_scores_{dataset_key}.png"
    save_score_chart(delta_df, results, threshold,
                     label_fn=lambda _: sample_label,
                     title=f"Drug Functional Resistance Profile — {description}",
                     out_path=out_path, target_lbl=target_lbl)


# ── GSE3926 pipeline ──────────────────────────────────────────────────────────

def run_gse3926(cfg):
    print("=== Initializing (GSE3926 — same-platform) ===")
    ensemble_models    = joblib.load(MODELS_FILE)
    ensemble_selectors = joblib.load(SELECTORS_FILE)
    disambig_model     = joblib.load(DISAMBIG_MODEL)
    disambig_selector  = joblib.load(DISAMBIG_SEL)
    full_genes         = open(GENE_LIST_FILE).read().splitlines()
    X_train_ref        = joblib.load(TRAIN_REF_FILE)

    meta = pd.read_csv(cfg["meta_file"]).set_index("Sample_ID")
    meta["CellLine"] = meta["Title"].apply(extract_cell_line)

    valid_ids     = meta[meta["Condition"].isin(cfg["valid_conditions"])].index.tolist()
    print(f"  Valid samples matched: {len(valid_ids)}")

    all_data = clean_columns(pd.read_csv(cfg["expr_file"], index_col=0))
    new_data = log2_if_raw(all_data.loc[valid_ids].copy())
    new_data = zscore_columns(new_data)

    untreated_meta    = meta[(meta["Condition"] == "untreated") & meta.index.isin(valid_ids)]
    # Target both typical treated and resistant labeled samples for parsing
    treated_meta      = meta[meta["Condition"].isin(["treated_dox", "resistant"]) & meta.index.isin(valid_ids)]
    
    untreated_by_line = (
        untreated_meta.groupby("CellLine", group_keys=False)
        .apply(lambda g: g.index[0], include_groups=False)
        .to_dict()
    )

    delta_rows  = {}
    paired_meta = {}
    for sid, row in treated_meta.iterrows():
        partner = untreated_by_line.get(row["CellLine"])
        if partner and partner in new_data.index and sid in new_data.index:
            delta_rows[sid]  = new_data.loc[sid] - new_data.loc[partner]
            paired_meta[sid] = {"cell_line": row["CellLine"], "partner_id": partner}

    delta_df = pd.DataFrame(delta_rows).T
    print(f"  Generated {len(delta_df)} expression response delta vectors.")

    delta_aligned = delta_df.reindex(columns=full_genes, fill_value=0.0)
    results = run_two_stage(delta_aligned, ensemble_models, ensemble_selectors,
                            disambig_model, disambig_selector, cfg["detection_threshold"])
    
    print_results_multi(delta_aligned, results, meta, paired_meta,
                        cfg["detection_threshold"], cfg, dataset_key="gse3926")


# ── GSE69644 pipeline ─────────────────────────────────────────────────────────

def run_gse69644(cfg):
    print("=== Initializing (GSE69644 — cross-platform) ===")
    ensemble_models    = joblib.load(MODELS_FILE)
    ensemble_selectors = joblib.load(SELECTORS_FILE)
    disambig_model     = joblib.load(DISAMBIG_MODEL)
    disambig_selector  = joblib.load(DISAMBIG_SEL)
    X_train_ref        = joblib.load(TRAIN_REF_FILE)
    full_genes         = open(GENE_LIST_FILE).read().splitlines()

    annot_ok = os.path.exists(GPL96_ANNOT) and os.path.exists(GPL13667_ANNOT)
    meta = pd.read_csv(cfg["meta_file"]).set_index("Sample_ID")
    valid_ids    = meta[meta["Condition"].isin(cfg["valid_conditions"])].index.tolist()

    all_data = clean_columns(pd.read_csv(cfg["expr_file"], index_col=0))
    new_data = all_data.loc[valid_ids].copy().astype(float)

    if annot_ok:
        map_69644      = load_probe_to_gene(GPL13667_ANNOT)
        map_96         = load_probe_to_gene(GPL96_ANNOT)

        print(f"  [DEBUG] Total Probe ID awal pada GPL13667: {len(map_69644)}")
        print(f"  [DEBUG] Total Probe ID setelah Variance Threshold (Latih): {X_train_ref.shape[1]}")

        new_data_genes = collapse_to_gene_symbols(new_data, map_69644)
        X_train_genes  = collapse_to_gene_symbols(X_train_ref, map_96)
        common         = list(set(new_data_genes.columns) & set(X_train_genes.columns))

        print(f"  [DEBUG] Jumlah Gene Symbol hasil irisan (Common Genes): {len(common)}")

        new_data_aligned    = new_data_genes[common]
        X_train_ref_aligned = X_train_genes[common]
        feature_label       = "gene symbols"
    else:
        new_data_aligned    = new_data
        X_train_ref_aligned = X_train_ref
        map_96              = {}
        feature_label       = "probe IDs fallback"

    new_data_aligned    = zscore_rows(new_data_aligned.astype(float))
    X_train_ref_aligned = zscore_rows(X_train_ref_aligned.astype(float))

    ctrl_ids    = meta[(meta.index.isin(valid_ids)) & (meta["Condition"] == "untreated")].index.tolist()
    treated_ids = meta[(meta.index.isin(valid_ids)) & (meta["Condition"] == "treated_cis_24h")].index.tolist()

    control_mean = new_data_aligned.loc[ctrl_ids].mean(axis=0)
    treated_mean = new_data_aligned.loc[treated_ids].mean(axis=0)

    if annot_ok:
        harmonized   = apply_combat(X_train_ref_aligned, new_data_aligned)
        control_mean = harmonized.loc[ctrl_ids].mean(axis=0)
        treated_mean = harmonized.loc[treated_ids].mean(axis=0)

    delta = (treated_mean - control_mean).to_frame(name="HK2_cisplatin_24h").T

    if annot_ok:
        probe_vals = {p: delta[map_96[p]].iloc[0]
                      if map_96.get(p) and map_96[p] in delta.columns else 0.0
                      for p in full_genes}
        delta_aligned = pd.DataFrame([probe_vals], columns=full_genes, index=["HK2_cisplatin_24h"])
    else:
        delta_aligned = delta.reindex(columns=full_genes, fill_value=0.0)

    results = run_two_stage(delta_aligned, ensemble_models, ensemble_selectors,
                            disambig_model, disambig_selector, cfg["detection_threshold"])
    
    print_results_single(delta_aligned, results, "HK-2 Cells",
                         threshold=cfg["detection_threshold"],
                         description="GSE69644 (HK-2 Kidney Epithelial Cells, 24h Cisplatin)",
                         cfg=cfg, dataset_key="gse69644")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <GSExxx_series_matrix.txt>")
        sys.exit(1)

    input_file  = sys.argv[1]
    dataset_key = detect_dataset(input_file)

    if dataset_key is None:
        print(f"ERROR: Could not detect dataset from '{input_file}'.")
        sys.exit(1)

    cfg = DATASET_CONFIGS[dataset_key]
    print(f"Dataset  : {cfg['description']}")
    print(f"Threshold: {cfg['detection_threshold']}\n")

    if dataset_key == "gse3926":
        run_gse3926(cfg)
    elif dataset_key == "gse69644":
        run_gse69644(cfg)


if __name__ == "__main__":
    main()
