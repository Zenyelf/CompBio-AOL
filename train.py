import pandas as pd
import gzip
import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "dataset"

drug_config = {
    1: {"name": "cisplatin",   "file": "GSE116439_series_matrix.txt.gz"},
    2: {"name": "doxorubicin", "file": "GSE116441_series_matrix.txt.gz"},
    3: {"name": "lapatinib",   "file": "GSE116445_series_matrix.txt.gz"},
    4: {"name": "vorinostat",  "file": "GSE116451_series_matrix.txt.gz"},
    5: {"name": "sunitinib",   "file": "GSE116449_series_matrix.txt.gz"},
}

def log2_if_raw(df):
    if df.max().max() > 50:
        return np.log2(df.astype(float) + 1)
    return df.astype(float)

def zscore_columns(df):
    # Mathematically standardizes each gene (column) across all samples
    return (df - df.mean()) / (df.std() + 1e-8)

# ─── 1. Load all datasets & Build Delta Vectors ───────────────────────────────

all_X       = []
all_y_multi = []

for label, info in drug_config.items():
    print(f"Loading {info['name']}...")
    full_path = os.path.join(DATA_DIR, info["file"])
    if not os.path.exists(full_path):
        print(f"  -> Not found, skipping.")
        continue

    titles     = []
    start_line = 0
    with gzip.open(full_path, "rt") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("!Sample_title"):
                titles = [t.strip().strip('"') for t in line.split("\t")[1:]]
            if line.startswith("!series_matrix_table_begin"):
                start_line = i + 1
                break

    df = pd.read_csv(full_path, sep="\t", skiprows=start_line,
                     index_col=0, compression="gzip")
    if df.index[-1] == "!series_matrix_table_end":
        df = df.drop(df.index[-1])

    columns_to_keep = [gsm_id for gsm_id, title in zip(df.columns, titles) if "_24h" in title]
    df_filtered = df[columns_to_keep].T
    df_filtered = log2_if_raw(df_filtered)
    
    # Apply column-wise Z-scoring
    df_filtered = zscore_columns(df_filtered)

    sample_info = []
    for gsm_id, title in zip(df.columns, titles):
        if "_24h" in title:
            cell_line = title.split("_")[0]
            is_control = "_0nM_" in title
            sample_info.append({"gsm_id": gsm_id, "cell_line": cell_line, "is_control": is_control})

    sample_meta = pd.DataFrame(sample_info).set_index("gsm_id")
    controls = sample_meta[sample_meta["is_control"]]
    control_by_line = controls.groupby("cell_line").apply(lambda x: x.index[0], include_groups=False).to_dict()
    treated = sample_meta[~sample_meta["is_control"]]

    delta_rows_train = {}
    labels_for_file = []
    for sid, row in treated.iterrows():
        cline = row["cell_line"]
        ctrl_sid = control_by_line.get(cline)
        if ctrl_sid and ctrl_sid in df_filtered.index and sid in df_filtered.index:
            delta_rows_train[sid] = df_filtered.loc[sid] - df_filtered.loc[ctrl_sid]
            labels_for_file.append(label)

    if delta_rows_train:
        df_deltas = pd.DataFrame(delta_rows_train).T
        all_X.append(df_deltas)
        all_y_multi.extend(labels_for_file)
        print(f"  -> Generated {len(delta_rows_train)} paired delta samples")

X_final = pd.concat(all_X, axis=0).fillna(0)
y_multi = pd.Series(all_y_multi, name="Multi_Label", index=X_final.index)

print(f"\n=== Original Dataset: {X_final.shape[0]} samples × {X_final.shape[1]} genes ===")

# ─── NEW: Variance Threshold (Kill dead/noisy probes) ─────────────────────────
# This removes genes that have near-zero variance (flat lines) across samples
var_selector = VarianceThreshold(threshold=0.05)
X_final_filtered = var_selector.fit_transform(X_final)
retained_genes = X_final.columns[var_selector.get_support()]
X_final = pd.DataFrame(X_final_filtered, index=X_final.index, columns=retained_genes)

print(f"=== Filtered Dataset: {X_final.shape[0]} samples × {X_final.shape[1]} genes (High Variance) ===")
print(y_multi.value_counts().sort_index().to_string())

joblib.dump(X_final, "training_reference_matrix.pkl")
print("\nSaved: training_reference_matrix.pkl")

# ─── 2. Train/test split ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# ─── 3. Stage 1 — Binary detection ensemble ───────────────────────────────────
print("\n=== Stage 1: Binary Detection Ensemble ===")
ensemble_models    = {}
ensemble_selectors = {}

for label, info in drug_config.items():
    drug_name      = info["name"]
    y_train_binary = (y_train == label).astype(int)
    y_test_binary  = (y_test  == label).astype(int)

    # Tightened from 300 to 200 to force focus on core genes
    selector    = SelectKBest(score_func=f_classif, k=200)
    X_train_sel = selector.fit_transform(X_train, y_train_binary)
    X_test_sel  = selector.transform(X_test)

    # NEW: L1 Regularization (Lasso) via liblinear
    # This forces the model to assign a weight of exactly 0.0 to generic stress genes
    model = LogisticRegression(
        penalty="l1", C=0.5, class_weight="balanced",
        max_iter=2000, solver="liblinear", random_state=42,
    )
    model.fit(X_train_sel, y_train_binary)

    score = model.score(X_test_sel, y_test_binary)
    print(f"  {drug_name:<14}  binary accuracy: {score:.3f}")

    ensemble_models[label]    = model
    ensemble_selectors[label] = selector

# ─── 4. Stage 2 — Multi-class disambiguation model ────────────────────────────
print("\n=== Stage 2: Multi-class Disambiguation Model ===")

# Tightened from 500 to 150 to force the Random Forest to only use highly specific markers
disambig_selector = SelectKBest(score_func=f_classif, k=150)
X_train_disambig  = disambig_selector.fit_transform(X_train, y_train)
X_test_disambig   = disambig_selector.transform(X_test)

disambig_model = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
)
disambig_model.fit(X_train_disambig, y_train)

score = disambig_model.score(X_test_disambig, y_test)
print(f"  Drug-vs-drug accuracy (test): {score:.3f}")

# ─── 5. Save all artifacts ────────────────────────────────────────────────────
print("\n=== Saving ===")
joblib.dump(ensemble_models,    "Isolated_Ensemble_Models.pkl")
joblib.dump(ensemble_selectors, "Isolated_Ensemble_Selectors.pkl")
joblib.dump(disambig_model,     "Disambiguation_Model.pkl")
joblib.dump(disambig_selector,  "Disambiguation_Selector.pkl")

with open("isolated_gene_list.txt", "w") as f:
    f.write("\n".join(X_train.columns.tolist()))
print("Done re-training on delta vectors.")
