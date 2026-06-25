import io
import os
import sys
import pandas as pd


# ── Per-dataset configuration ─────────────────────────────────────────────────

DATASET_CONFIGS = {
    "gse3926": {
        "description":      "GSE3926 — GPL96 (HGU133A), doxorubicin study",
        "already_log2":     False,
        "valid_conditions": ["untreated", "treated_dox", "resistant"],  # INCLUDED resistant samples
        "excluded_tag":     "none",
        "excluded_note": (
            "NOTE: Resistant samples are now INCLUDED in the pipeline.\n"
            "      predict.py will evaluate if they show a flat (resistant) signature."
        ),
    },
    "gse69644": {
        "description":      "GSE69644 — GPL13667 (HG-U219), cisplatin study, HK-2 cells",
        "already_log2":     True,
        "valid_conditions": ["untreated", "treated_cis_24h"],
        "excluded_tag":     "treated_cis_6h",
        "excluded_note": (
            "NOTE: 'treated_cis_6h' samples will be excluded — training used 24h exposure only.\n"
            "NOTE: Platform is GPL13667 (HG-U219), different from training (GPL96 HGU133A).\n"
            "      predict.py will map probe IDs → gene symbols before predicting."
        ),
    },
}


# ── Condition taggers ─────────────────────────────────────────────────────────

def _tag_gse3926(title):
    t = title.lower()
    if "_rdb" in t or "rdb" in t:
        return "resistant"
    elif "_treatdb" in t or "treatdb" in t:
        return "treated_dox"
    elif "_par" in t or "par" in t:
        return "untreated"
    return "unknown"


def _tag_gse69644(title):
    t = title.lower()
    if "control" in t or "non treated" in t or "untreated" in t:
        return "untreated"
    elif "24h" in t or "24 h" in t:
        return "treated_cis_24h"
    elif "6h" in t or "6 h" in t:
        return "treated_cis_6h"
    return "unknown"


TAGGERS = {
    "gse3926":  _tag_gse3926,
    "gse69644": _tag_gse69644,
}


# ── Auto-detect dataset from filename ────────────────────────────────────────

def detect_dataset(filepath):
    name = os.path.basename(filepath).lower()
    for key in DATASET_CONFIGS:
        if key in name:
            return key
    return None


# ── GEO Series Matrix parser ──────────────────────────────────────────────────

def parse_series_matrix(filepath):
    metadata_raw = {}
    data_lines   = []
    in_matrix    = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line     = line.rstrip("\n")
            stripped = line.strip()

            if stripped == "!series_matrix_table_begin":
                in_matrix = True
                continue
            elif stripped == "!series_matrix_table_end":
                in_matrix = False
                continue

            if in_matrix:
                data_lines.append(stripped)
            elif line.startswith("!Sample_geo_accession") or line.startswith("!Sample_title"):
                tag    = line.split("\t")[0].lstrip("!")
                values = [v.strip().strip('"') for v in line.split("\t")[1:]]
                metadata_raw[tag] = values

    if not data_lines:
        raise ValueError(f"'!series_matrix_table_begin' not found in {filepath}")

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t")
    df.columns = [c.replace('"', "").strip() for c in df.columns]
    first = df.columns[0]
    if first in ("!ID_REF", "ID_REF"):
        df.rename(columns={first: "ID_REF"}, inplace=True)
    df.set_index("ID_REF", inplace=True)

    df_ml            = df.T
    df_ml.index.name = "Sample_ID"
    return df_ml, metadata_raw


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert.py <GSExxx_series_matrix.txt>")
        sys.exit(1)

    input_file  = sys.argv[1]
    dataset_key = detect_dataset(input_file)

    if dataset_key is None:
        print(f"ERROR: Could not detect dataset from filename '{input_file}'.")
        print(f"       Filename must contain one of: {', '.join(DATASET_CONFIGS)}")
        sys.exit(1)

    cfg    = DATASET_CONFIGS[dataset_key]
    tagger = TAGGERS[dataset_key]

    expr_out    = f"new_samples_{dataset_key}.csv"
    meta_out    = f"sample_metadata_{dataset_key}.csv"

    print(f"Dataset  : {cfg['description']}")
    print(f"Input    : {input_file}")

    df_ml, metadata_raw = parse_series_matrix(input_file)

    # ── Expression CSV ────────────────────────────────────────────────────────
    df_ml.to_csv(expr_out)
    vmin, vmax = df_ml.values.min(), df_ml.values.max()
    print(f"\nExpression data  → {expr_out}")
    print(f"  {df_ml.shape[0]} samples × {df_ml.shape[1]} probes")
    print(f"  Value range: {vmin:.3f} – {vmax:.3f}", end="")
    if cfg["already_log2"]:
        print("  (already log2/RMA — NO transform needed)")
    else:
        print("  (raw — log2 transform applied in predict.py)")

    # ── Metadata CSV ──────────────────────────────────────────────────────────
    gsm_ids = metadata_raw.get("Sample_geo_accession", df_ml.index.tolist())
    titles  = metadata_raw.get("Sample_title", [""] * len(gsm_ids))

    meta = pd.DataFrame({
        "Sample_ID": gsm_ids,
        "Title":     titles,
        "Condition": [tagger(t) for t in titles],
    })
    meta.to_csv(meta_out, index=False)

    print(f"\nSample metadata  → {meta_out}\n")
    print(meta.to_string(index=False))
    print()
    print(cfg["excluded_note"])

    counts = meta["Condition"].value_counts()
    print("\nCondition counts:")
    for cond, n in counts.items():
        tag = " ← valid" if cond in cfg["valid_conditions"] else " ← excluded"
        print(f"  {cond:<20} {n:>3}{tag}")


if __name__ == "__main__":
    main()
