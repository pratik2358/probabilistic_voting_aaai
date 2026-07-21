"""
Strict CSV parser for real cast-vote-record data used in the
run_real_elections pipeline.

Design contract:
- Reads ballots only from what is in the file. No synthetic generation.
- Recognizes four header families (NYC 2023, NYC 2021, countingGroup, tally_type)
  by exact column-presence. Hard-fails on any other header layout.
- Classifies every row as bound or unbound by an exact value lookup.
  Hard-fails on any value not in the lookup table.
- Bound      = early voting + election day + provisional / affidavit
- Unbound    = absentee + mail + question (per project rule)

Output is a base_parser.ElectionData identical in shape to what the prior
EnhancedCSVParser produced, so run_real_elections.py and
experiment_real_ballots_setup do not need any changes.
"""

import csv
import os
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

from .base_parser import BaseParser, ElectionData


# ── rank-cell tokens that mean "no candidate ranked here" ─────────────────────
SKIP_TOKENS: Set[str] = {
    "", "skipped", "overvote", "undervote",
    "Write-in", "write-in", "writein", "WRITEIN",
}


# ── categories that lump VBM and EV into a single label ───────────────────────
# Tagged in metadata['has_merged_vbm_ev']; treated as
# unbound (vote-by-mail dominant in the SF jurisdictions where this label
# appears).
MERGED_VBM_EV_CATEGORIES: Set[str] = {
    "VBM / EV - 400c 1st Cut",
    "VBM / EV - 400c 2nd Cut",
}


# ── column 'countingGroup' (Alaska + post-2019 SF/Oakland/Berkeley + LasCruces) ─
COUNTING_GROUP_MAP: Dict[str, str] = {
    # election day
    "Election Day":   "bound",
    "Election Night": "bound",   # Oakland 2020 uses this label for election-day batch
    # early voting
    "Early Voting":   "bound",   # Alaska
    "Early":          "bound",   # LasCruces
    # absentee / mail
    "Absentee":       "unbound", # Alaska, LasCruces
    "Vote by Mail":   "unbound", # post-2019 SF/Oakland/Berkeley
    # provisional / question
    "Question":       "unbound", # Alaska
    "EV Prov":        "bound",   # LasCruces — early-voting provisional (provisional → bound)
}


# ── column 'tally_type' (legacy SF/Oakland/Berkeley/SanLeandro/PierceCounty) ──
TALLY_TYPE_MAP: Dict[str, str] = {
    # legacy SF/Oakland/Berkeley/SanLeandro
    "Election Day - 400C":      "bound",
    "Election Day - Insight":   "bound",
    "Election Day - Edge Dups": "bound",
    "Provisional - 400C":       "bound",   # provisional → bound per project rule
    "Vote by Mail - 400C":      "unbound",
    "MVBM - 400C":              "unbound",
    "VBM / EV - 400c 1st Cut":  "unbound", # see MERGED_VBM_EV_CATEGORIES
    "VBM / EV - 400c 2nd Cut":  "unbound",
    # PierceCounty 2008
    "Absentee Voting":          "unbound",
    "Early vote":               "bound",
    "Early vote Provisional":   "bound",   # provisional → bound per project rule
    "Precinct Voting":          "bound",
    "Regular Poll Ballots":     "bound",
}


# ── column 'Type' (NYC 2023) ──────────────────────────────────────────────────
NYC_2023_TYPE_MAP: Dict[str, str] = {
    "EAR": "bound",   # early voting
    "ELE": "bound",   # election day
    "OTH": "unbound", # other (absentee + affidavit + emergency lumped)
}


# ── column 'source_file' (NYC 2021 DEM) ───────────────────────────────────────
# Values look like 2021P3V1_ABS, 2021P4V1_ABS_DEM, 2021P3V1_AFF,
# 2021P3V1_ELE1, 2021P3V1_EMG, 2021PnVk_OTH, etc.
NYC_2021_TOKEN_PATTERN = re.compile(
    r"_(ABS|AFF|EMG|OTH|ELE\d*)(?:_[A-Z]+)?$"
)
NYC_2021_TOKEN_MAP: Dict[str, str] = {
    "ABS": "unbound",   # absentee
    "AFF": "bound",     # affidavit (NY's term for provisional) → bound per project rule
    "OTH": "unbound",   # other / catch-all
    "ELE": "bound",     # election day
    "EMG": "bound",     # emergency (election-day backup)
}


def _classify_nyc_2021(value: str) -> Optional[str]:
    """Return 'bound'/'unbound' for an NYC-2021 source_file value, else None."""
    m = NYC_2021_TOKEN_PATTERN.search(value)
    if not m:
        return None
    token = m.group(1)
    canonical = "ELE" if token.startswith("ELE") else token
    return NYC_2021_TOKEN_MAP.get(canonical)


# ── parser ────────────────────────────────────────────────────────────────────
class StrictRealBallotParser(BaseParser):
    """
    Hard-failing parser for real cast-vote-record CSVs.
    """

    def can_parse(self, filepath: str) -> bool:
        return filepath.endswith(".csv")

    def parse(self, filepath: str, **kwargs) -> ElectionData:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        # Most files are UTF-8; a few (e.g. older Berkeley) are Latin-1 with
        # accented candidate names. Try UTF-8 strict first, fall back to
        # Latin-1 (single-byte, never raises) on a decode error.
        encoding_used = "utf-8"
        try:
            with open(filepath, "r", encoding="utf-8", errors="strict", newline="") as f:
                rows = list(csv.reader(f))
        except UnicodeDecodeError:
            encoding_used = "latin-1"
            with open(filepath, "r", encoding="latin-1", newline="") as f:
                rows = list(csv.reader(f))

        if len(rows) < 2:
            raise ValueError(f"CSV is empty or has no data rows: {filepath}")

        header = rows[0]
        data_rows = rows[1:]

        format_name, classifier_col, classifier_fn = self._detect_format(header)
        classifier_idx = header.index(classifier_col)

        rank_indices = [
            i for i, h in enumerate(header)
            if h.lower().startswith("rank") or h.lower().startswith("choice")
        ]
        if not rank_indices:
            raise ValueError(
                f"No rank columns found in {filepath}. Header: {header}"
            )

        bound_ballots: List[List[str]] = []
        unbound_ballots: List[List[str]] = []
        category_counts: Dict[str, Dict[str, int]] = {}
        all_candidates: Set[str] = set()
        unknown_values: Dict[str, int] = {}

        for row_idx, row in enumerate(data_rows):
            if not row or all(not c for c in row):
                continue
            if classifier_idx >= len(row):
                raise ValueError(
                    f"Row {row_idx + 2} in {filepath} has fewer fields than "
                    f"the header expects ({len(row)} vs {len(header)})"
                )

            raw_value = row[classifier_idx].strip()
            classification = classifier_fn(raw_value)

            if classification not in ("bound", "unbound"):
                unknown_values[raw_value] = unknown_values.get(raw_value, 0) + 1
                continue

            ranking: List[str] = []
            for i in rank_indices:
                if i < len(row):
                    val = row[i].strip()
                    if val and val not in SKIP_TOKENS:
                        ranking.append(val)

            if not ranking:
                continue  # blank ballot

            all_candidates.update(ranking)
            (bound_ballots if classification == "bound" else unbound_ballots).append(ranking)

            cat = category_counts.setdefault(raw_value, {"bound": 0, "unbound": 0})
            cat[classification] += 1

        if unknown_values:
            samples = "\n".join(
                f"  {v!r} ({n} rows)"
                for v, n in sorted(unknown_values.items(), key=lambda kv: -kv[1])
            )
            raise ValueError(
                f"Unknown ballot-type values in {filepath}\n"
                f"  format = {format_name}\n"
                f"  column = {classifier_col!r}\n"
                f"Values not in mapping:\n{samples}\n"
                f"Add them to the parser's mapping or fix the data."
            )

        if not bound_ballots:
            raise ValueError(
                f"No bound ballots in {filepath} (format={format_name})"
            )
        if not unbound_ballots:
            raise ValueError(
                f"No unbound ballots in {filepath} (format={format_name}). "
                f"This pipeline requires a real bound/unbound split — "
                f"synthetic generation is disabled."
            )

        candidates = sorted(all_candidates)
        if len(candidates) < 2:
            raise ValueError(
                f"Fewer than 2 candidates in {filepath}: {candidates}"
            )

        n_bound = len(bound_ballots)
        U = len(unbound_ballots)

        # Prior P: first-choice frequencies from bound ballots, computed by the
        # SAME function the experiment uses (BaseParser.compute_first_choice_probs)
        # so metadata P is identical to the P the methods receive. Candidates
        # with no bound first-choice support get an exact 0 — deliberately
        # un-floored: the methods handle it exactly (exact/Mult skip zero-prob
        # allocations, FFT places no mass on that axis, MVN uses
        # allow_singular + jitter).
        P = self.compute_first_choice_probs(bound_ballots, candidates)

        ballot_classification = {
            cat: classifier_fn(cat) or "unknown" for cat in category_counts
        }
        has_merged_vbm_ev = any(
            cat in MERGED_VBM_EV_CATEGORIES for cat in category_counts
        )

        election_name = os.path.basename(filepath).replace(".csv", "")
        metadata = {
            "source_file":            filepath,
            "format":                 format_name,
            "encoding":               encoding_used,
            "classifier_column":      classifier_col,
            "ballot_classification":  ballot_classification,
            "category_counts":        category_counts,
            "has_merged_vbm_ev":      has_merged_vbm_ev,
            "n_total":                n_bound + U,
            "n_bound":                n_bound,
            "n_unbound":              U,
            "n_candidates":           len(candidates),
        }

        return ElectionData(
            name=election_name,
            candidates=candidates,
            bound_ballots=bound_ballots,
            n_bound=n_bound,
            U=U,
            P=P,
            metadata=metadata,
            unbound_ballots=unbound_ballots,
        )

    # ── format detection ──────────────────────────────────────────────────────
    def _detect_format(
        self, header: List[str]
    ) -> Tuple[str, str, Callable[[str], Optional[str]]]:
        col_set = set(header)

        if "Cast.Vote.Record" in col_set and "Type" in col_set:
            return ("nyc_2023_type", "Type",
                    lambda v: NYC_2023_TYPE_MAP.get(v.strip()))

        if "source_file" in col_set and "Ballot.Style" in col_set:
            return ("nyc_2021_source_file", "source_file",
                    lambda v: _classify_nyc_2021(v.strip()))

        if "countingGroup" in col_set:
            return ("counting_group", "countingGroup",
                    lambda v: COUNTING_GROUP_MAP.get(v.strip()))

        if "tally_type" in col_set:
            return ("legacy_tally_type", "tally_type",
                    lambda v: TALLY_TYPE_MAP.get(v.strip()))

        raise ValueError(
            f"Unrecognized CSV header: {header}\n"
            f"Expected one of these column markers:\n"
            f"  - 'Type' + 'Cast.Vote.Record'   (NYC 2023)\n"
            f"  - 'source_file' + 'Ballot.Style' (NYC 2021 DEM)\n"
            f"  - 'countingGroup'                (Alaska / post-2019 SF/Oakland/Berkeley / LasCruces)\n"
            f"  - 'tally_type'                   (legacy SF/Oakland/Berkeley/SanLeandro/PierceCounty)"
        )


# ── self-test CLI ─────────────────────────────────────────────────────────────
def _selftest(filepath: str) -> int:
    parser = StrictRealBallotParser()
    try:
        ed = parser.parse(filepath)
    except (ValueError, FileNotFoundError) as e:
        print(f"FAIL: {e}")
        return 1

    print(f"file               : {filepath}")
    print(f"format             : {ed.metadata['format']}")
    print(f"classifier column  : {ed.metadata['classifier_column']}")
    print(f"candidates ({len(ed.candidates)}): {ed.candidates}")
    print(f"n_bound            : {ed.n_bound:,}")
    print(f"U (unbound)        : {ed.U:,}")
    print(f"has_merged_vbm_ev  : {ed.metadata['has_merged_vbm_ev']}")
    print()
    print("category breakdown:")
    for cat, counts in sorted(ed.metadata["category_counts"].items()):
        cls = ed.metadata["ballot_classification"][cat]
        total = counts["bound"] + counts["unbound"]
        print(f"  {cat!r:42s} → {cls:7s}  ({total:,} ballots)")
    print()
    print("first-choice prior P (top 10):")
    for c, p in sorted(ed.P.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {p:.4f}  {c}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m preprocessing.formats.strict_parser <election.csv>")
        sys.exit(2)
    sys.exit(_selftest(sys.argv[1]))
