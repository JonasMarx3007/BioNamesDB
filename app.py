from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="BioNameDB Translator", layout="centered")

UNIPROT_ALL_ACCESSIONS = "UniProt All Accessions"
VIRTUAL_COL_UNIPROT_ALL = "__uniprot_all_accessions__"

DB_TO_COL = {
    "HGNC ID": "hgnc_id",
    "HGNC Symbol": "symbol",
    "HGNC Name": "name",
    "HGNC Alias Symbol": "alias_symbol",
    "Ensembl": "ensembl_gene_id",
    "VEGA": "vega_id",
    "ENA": "ena",
    "RefSeq": "refseq_accession",
    "CCDS": "ccds_id",
    "PubMed": "pubmed_id",
    "Enzyme Commission (EC)": "enzyme_id",
    "RNAcentral": "rna_central_id",
    "UniProt Accession": "uniprot_symbol",
    "UniProt Name": "uniprot_name",
    "UniProt Gene": "uniprot_gene",
    "UniProt Description": "uniprot_description",
    "UniProt Secondary Accession": "uniprot_alternative",
    UNIPROT_ALL_ACCESSIONS: VIRTUAL_COL_UNIPROT_ALL,
}

SPLIT_SEMI = re.compile(r"\s*;\s*")


def split_cell_values(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = [p.strip() for p in SPLIT_SEMI.split(s) if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def parse_inputs(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"\s+", text.strip())
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def parse_text_rows(raw: str) -> list[str]:
    if not raw:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    tokens = re.split(r"\s+", raw)
    return [t.strip() for t in tokens if t and t.strip()]


@st.cache_data(show_spinner=False)
def load_db(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False).replace("", pd.NA)


@st.cache_data(show_spinner=False)
def load_user_table(uploaded_file) -> tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False).replace("", pd.NA)
        return df, "csv"
    df = pd.read_csv(uploaded_file, sep="\t", dtype=str, keep_default_na=False).replace("", pd.NA)
    return df, "tsv"


def _values_from_row(row: pd.Series, col_or_virtual: str) -> list[str]:
    if col_or_virtual == VIRTUAL_COL_UNIPROT_ALL:
        primary = split_cell_values(row.get("uniprot_symbol", pd.NA))
        secondary = split_cell_values(row.get("uniprot_alternative", pd.NA))
        seen, out = set(), []
        for t in primary + secondary:
            if t and t not in seen:
                out.append(t)
                seen.add(t)
        return out
    return split_cell_values(row.get(col_or_virtual, pd.NA))


@st.cache_data(show_spinner=False)
def build_index(db_df: pd.DataFrame, in_label: str, out_label: str) -> dict[str, list[str]]:
    in_col = DB_TO_COL[in_label]
    out_col = DB_TO_COL[out_label]
    idx: dict[str, set[str]] = defaultdict(set)

    for _, row in db_df.iterrows():
        ins = _values_from_row(row, in_col)
        outs = _values_from_row(row, out_col)
        if not ins or not outs:
            continue
        for i in ins:
            for o in outs:
                idx[i].add(o)

    return {k: sorted(v) for k, v in idx.items()}


@st.cache_data(show_spinner=False)
def build_value_sets(db_df: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for label in labels:
        col = DB_TO_COL[label]
        vals: set[str] = set()

        if col == VIRTUAL_COL_UNIPROT_ALL:
            if "uniprot_symbol" in db_df.columns:
                for v in db_df["uniprot_symbol"].dropna().astype(str):
                    for t in split_cell_values(v):
                        vals.add(t)
            if "uniprot_alternative" in db_df.columns:
                for v in db_df["uniprot_alternative"].dropna().astype(str):
                    for t in split_cell_values(v):
                        vals.add(t)
            out[label] = vals
            continue

        if col not in db_df.columns:
            out[label] = set()
            continue

        for v in db_df[col].dropna().astype(str):
            for t in split_cell_values(v):
                vals.add(t)

        out[label] = vals

    return out


def autodetect_input_db(ids: list[str], db_df: pd.DataFrame) -> tuple[str | None, dict[str, int]]:
    labels = list(DB_TO_COL.keys())
    col_values = build_value_sets(db_df, tuple(labels))

    scores: dict[str, int] = {}
    for label in labels:
        values = col_values.get(label, set())
        scores[label] = sum(1 for x in ids if x in values)

    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        return None, scores

    top_labels = [lab for lab, sc in scores.items() if sc == best_score]
    if UNIPROT_ALL_ACCESSIONS in top_labels:
        return UNIPROT_ALL_ACCESSIONS, scores

    for label in labels:
        if scores.get(label, 0) == best_score:
            return label, scores

    return None, scores


def translate_cell(cell: object, index: dict[str, list[str]]) -> tuple[str, bool]:
    toks = split_cell_values(cell)
    if not toks:
        return "NA", False

    out_vals = set()
    for t in toks:
        for o in index.get(t, []):
            out_vals.add(o)

    if not out_vals:
        return "NA", False

    return ";".join(sorted(out_vals)), True


st.title("BioNameDB ID Translator")
st.caption("Translate between identifier types using BioNamesDB.txt")

db_path = Path(__file__).resolve().parent / "BioNamesDB.txt"
if not db_path.exists():
    st.error(f"Could not find {db_path.name} in the app directory.")
    st.stop()

db_df = load_db(str(db_path))
labels = list(DB_TO_COL.keys())
default_out = labels.index("HGNC Symbol") if "HGNC Symbol" in labels else 0

st.header("Text input")

c1, c2 = st.columns(2)
with c1:
    input_db_text = st.selectbox("Translate FROM (text)", labels, index=1, key="input_db_text")
with c2:
    output_db_text = st.selectbox("Translate TO (text)", labels, index=default_out, key="output_db_text")

raw = st.text_area(
    "Input (each whitespace/newline-separated token is a row; tokens may contain semicolon-separated IDs)",
    height=160,
    placeholder="A0A024QZX5;A0A087X1N8;P35237\nA0A024R161;P50151",
    key="raw_text",
)

b1, b2 = st.columns([1, 1])
with b1:
    auto_text = st.button("Auto-detect input DB (text)", type="secondary", key="auto_text")
with b2:
    run_text = st.button("Translate (text)", type="primary", key="run_text")

if "autodetected_input_db_text" not in st.session_state:
    st.session_state["autodetected_input_db_text"] = None

if auto_text:
    rows_for_detect = parse_text_rows(raw)
    expanded = []
    for cell in rows_for_detect:
        expanded.extend(split_cell_values(cell))
    expanded = expanded[:20000]

    if not expanded:
        st.warning("Please enter at least one ID to auto-detect.")
    else:
        with st.spinner("Auto-detecting input database…"):
            best_label, _ = autodetect_input_db(expanded, db_df)
        st.session_state["autodetected_input_db_text"] = best_label
        if best_label is None:
            st.info("No matches found in any column; cannot auto-detect.")
        else:
            st.success(f"Auto-detected input DB: {best_label}")

if run_text:
    cells = parse_text_rows(raw)
    if not cells:
        st.warning("Please enter at least one token (separated by whitespace/newlines).")
    else:
        effective_in = st.session_state.get("autodetected_input_db_text") or input_db_text

        with st.spinner("Building mapping and translating…"):
            index = build_index(db_df, effective_in, output_db_text)

        translated_vals = []
        translated_flags = []
        for cell in cells:
            val, ok = translate_cell(cell, index)
            translated_vals.append(val)
            translated_flags.append(ok)

        translated_count = int(sum(translated_flags))
        total = int(len(cells))

        out_df_text = pd.DataFrame(
            {
                "input": cells,
                "output": translated_vals,
                "matched": translated_flags,
            }
        )

        st.subheader(f"Output ({effective_in} → {output_db_text})")
        st.success(f"Translated: {translated_count}/{total} rows")
        st.dataframe(out_df_text, use_container_width=True)

        out_tsv = "\n".join([f"{a}\t{b}" for a, b in zip(cells, translated_vals)])
        st.download_button(
            "Download results (TSV)",
            data=out_tsv + "\n",
            file_name=f"{effective_in}_to_{output_db_text}.tsv".replace(" ", "_"),
            mime="text/tab-separated-values",
            key="dl_text",
        )

st.divider()

st.header("File upload (append new column)")

uploaded = st.file_uploader("Upload a table (.tsv / .csv / .txt)", type=["tsv", "csv", "txt"])

if "autodetected_input_db_file" not in st.session_state:
    st.session_state["autodetected_input_db_file"] = None

if uploaded is not None:
    user_df, user_fmt = load_user_table(uploaded)
    st.write(f"Loaded **{uploaded.name}** with **{len(user_df):,}** rows and **{len(user_df.columns):,}** columns.")
    col_pick = st.selectbox("Select the column to translate", list(user_df.columns), key="user_col_pick")

    f1, f2 = st.columns(2)
    with f1:
        input_db_file = st.selectbox("Translate FROM (file)", labels, index=1, key="input_db_file")
    with f2:
        output_db_file = st.selectbox("Translate TO (file)", labels, index=default_out, key="output_db_file")

    fb1, fb2 = st.columns([1, 1])
    with fb1:
        auto_file = st.button("Auto-detect input DB (file)", type="secondary", key="auto_file")
    with fb2:
        run_file = st.button("Translate & append column", type="primary", key="run_file")

    if auto_file:
        sample_vals = (
            user_df[col_pick]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s != ""]
            .head(5000)
            .tolist()
        )
        expanded: list[str] = []
        for v in sample_vals:
            expanded.extend(split_cell_values(v))
        expanded = expanded[:20000]

        if not expanded:
            st.warning("Selected column has no usable values to auto-detect.")
        else:
            with st.spinner("Auto-detecting input database…"):
                best_label, _ = autodetect_input_db(expanded, db_df)
            st.session_state["autodetected_input_db_file"] = best_label
            if best_label is None:
                st.info("No matches found in any column; cannot auto-detect.")
            else:
                st.success(f"Auto-detected input DB: {best_label}")

    if run_file:
        effective_in = st.session_state.get("autodetected_input_db_file") or input_db_file

        with st.spinner("Building mapping and translating…"):
            index = build_index(db_df, effective_in, output_db_file)

        new_col_name = f"{col_pick}__{effective_in}_to_{output_db_file}".replace(" ", "_")
        out_df = user_df.copy()

        translated_vals = []
        translated_flags = []
        for cell in out_df[col_pick].tolist():
            val, ok = translate_cell(cell, index)
            translated_vals.append(val)
            translated_flags.append(ok)

        out_df[new_col_name] = translated_vals

        translated_count = int(sum(translated_flags))
        total_rows = int(len(out_df))
        st.success(f"Translated: {translated_count}/{total_rows} rows")

        st.subheader("Preview")
        st.dataframe(out_df.head(50), use_container_width=True)

        if user_fmt == "csv":
            data = out_df.to_csv(index=False)
            mime = "text/csv"
            fname = uploaded.name.rsplit(".", 1)[0] + "_translated.csv"
        else:
            data = out_df.to_csv(sep="\t", index=False)
            mime = "text/tab-separated-values"
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            ext = ext if ext in ("tsv", "txt") else "tsv"
            fname = uploaded.name.rsplit(".", 1)[0] + f"_translated.{ext}"

        st.download_button(
            "Download file with appended column",
            data=data,
            file_name=fname,
            mime=mime,
            key="dl_file",
        )
else:
    st.info("Upload a file to enable table translation.")
