import sys
import time
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

def _ensure_packages() -> None:
    pkgs = ["pandas", "requests", "tqdm", "biopython"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

try:
    import pandas as pd
    import requests
    from tqdm import tqdm
    from Bio import SeqIO
except Exception:
    _ensure_packages()
    import pandas as pd
    import requests
    from tqdm import tqdm
    from Bio import SeqIO

FASTA_FILENAME = "uniprotkb_proteome_Homo_sapiens_ver20240503.fasta"
OUTPUT_FILENAME = "UniportDB.txt"

_ACC6 = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])$")
_ACC10 = re.compile(r"^[A-NR-Z0-9]{10}$")

def _is_accession(x: str) -> bool:
    x = (x or "").strip()
    return bool(_ACC6.match(x) or _ACC10.match(x))

def _parse_uniprot_json(j: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    gene = None
    genes = j.get("genes") or []
    if genes:
        gn = (genes[0] or {}).get("geneName") or {}
        gene = gn.get("value")

    desc = None
    pdsc = j.get("proteinDescription") or {}
    rec = pdsc.get("recommendedName") or {}
    full = rec.get("fullName") or {}
    desc = full.get("value")
    if not desc:
        subs = pdsc.get("submissionNames") or []
        if subs:
            sfull = (subs[0] or {}).get("fullName") or {}
            desc = sfull.get("value")

    sec = j.get("secondaryAccessions") or []
    sec_joined = ";".join([s for s in sec if s]) or None

    return gene, desc, sec_joined

def _fetch_by_accession(session: requests.Session, acc: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None, None, None
    return _parse_uniprot_json(r.json())

def _fetch_by_entry_name(session: requests.Session, entry_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": f"id:{entry_name}", "format": "json", "size": 1}
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None, None, None
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None, None, None
    return _parse_uniprot_json(results[0])

def append_uniprot_annotations(
    df: pd.DataFrame,
    symbol_col: str = "uniprot_symbol",
    name_col: str = "uniprot_name",
    sleep_s: float = 0.0,
    retries: int = 2,
) -> pd.DataFrame:
    out = df.copy()

    if symbol_col not in out.columns or name_col not in out.columns:
        raise ValueError(f"Input df must contain '{symbol_col}' and '{name_col}' columns.")

    def pick_key(row) -> Tuple[str, str]:
        acc = str(row.get(symbol_col) or "").strip()
        nm = str(row.get(name_col) or "").strip()
        if acc and _is_accession(acc):
            return ("accession", acc)
        if nm:
            return ("entry_name", nm)
        return ("none", "")

    keys = out.apply(pick_key, axis=1)
    unique_keys = [(k, v) for (k, v) in keys.tolist() if k != "none" and v]
    unique_keys = list(dict.fromkeys(unique_keys))

    cache: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str], Optional[str]]] = {}

    with requests.Session() as session:
        session.headers.update({"User-Agent": "pandas-uniprot-annotator/1.0"})

        pbar = tqdm(
            total=len(unique_keys),
            unit="entry",
            file=sys.stderr,
            leave=False,
            dynamic_ncols=False,
            ascii=True,
            mininterval=0.5,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for _, (k, v) in enumerate(unique_keys, start=1):
            for attempt in range(retries + 1):
                try:
                    if k == "accession":
                        cache[(k, v)] = _fetch_by_accession(session, v)
                    else:
                        cache[(k, v)] = _fetch_by_entry_name(session, v)
                    break
                except Exception:
                    if attempt < retries:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        cache[(k, v)] = (None, None, None)

            pbar.update(1)
            if sleep_s:
                time.sleep(sleep_s)

        pbar.close()

    genes, descs, alts = [], [], []
    for k, v in keys.tolist():
        if k == "none" or not v:
            genes.append(None)
            descs.append(None)
            alts.append(None)
            continue
        g, d, a = cache.get((k, v), (None, None, None))
        genes.append(g)
        descs.append(d)
        alts.append(a)

    out["uniprot_gene"] = genes
    out["uniprot_description"] = descs
    out["uniprot_alternative"] = alts

    return out

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fasta_path = base_dir / FASTA_FILENAME
    out_path = base_dir / OUTPUT_FILENAME

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found next to script: {fasta_path}")

    df = pd.DataFrame(
        {"id": rec.id, "description": rec.description, "sequence": str(rec.seq), "length": len(rec.seq)}
        for rec in SeqIO.parse(str(fasta_path), "fasta")
    )

    df = df[["id"]].copy()
    parts = df["id"].str.split("|", expand=True)

    df["uniprot_symbol"] = parts[1].astype(str).str.strip()
    df["uniprot_name"] = parts[2].astype(str).str.strip()
    df = df.drop(columns=["id"])

    df = append_uniprot_annotations(df)

    df.to_csv(out_path, sep="\t", index=False)

if __name__ == "__main__":
    main()
