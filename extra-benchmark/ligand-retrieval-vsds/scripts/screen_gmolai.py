#!/usr/bin/env python3
"""Label-blind gMolAI graph-contract screen without a model forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rdkit import Chem

from benchmark_io import (
    REPOSITORY_ROOT,
    atomic_write_json,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from gmolai_retrain.fast_graph import fast_featurize_molecule  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_panel_tsv(args.input)
    accepted: list[int] = []
    rejections: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:
                raise ValueError("RDKit parse failure")
            x, edge_index, edge_attr = fast_featurize_molecule(molecule)
            if x.shape[0] != molecule.GetNumAtoms() or edge_index.shape[0] != 2:
                raise RuntimeError("graph shape mismatch")
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise RuntimeError("edge shape mismatch")
            accepted.append(index)
        except Exception as error:
            rejections.append(
                {
                    "panel_index": index,
                    "molecule_hash": row["molecule_hash"],
                    "reason": f"{type(error).__name__}: {' '.join(str(error).split())}"[:1000],
                }
            )
    report = {
        "schema_version": 1,
        "status": "ok",
        "execution": "screen_only_no_model_forward_exact_fast_graph_contract",
        "model": "gmolai",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejections),
        "coverage_fraction": len(accepted) / max(1, len(rows)),
        "accepted_indices": accepted,
        "accepted_identity_sha256": sha256_lines(
            rows[index]["molecule_hash"] for index in accepted
        ),
        "rejections": rejections,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"accepted_indices", "rejections"}}, sort_keys=True))


if __name__ == "__main__":
    main()

